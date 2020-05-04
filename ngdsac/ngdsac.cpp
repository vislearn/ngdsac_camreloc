/*
Based on the DSAC++ and ESAC code.
https://github.com/vislearn/LessMore
https://github.com/vislearn/esac

Copyright (c) 2016, TU Dresden
Copyright (c) 2020, Heidelberg University
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of the TU Dresden, Heidelberg University nor the
      names of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL TU DRESDEN OR HEIDELBERG UNIVERSITY BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include <torch/extension.h>
#include <opencv2/opencv.hpp>

#include <iostream>

#include "thread_rand.h"
#include "stop_watch.h"

#include "ngdsac_types.h"
#include "ngdsac_util.h"
#include "ngdsac_loss.h"
#include "ngdsac_derivative.h"

#define MAX_REF_STEPS 100 // max pose refienment iterations

/**
 * @brief Estimate a camera pose based on a scene coordinate prediction, and neural guidance for RANSAC.
 * @param sceneCoordinatesSrc Scene coordinate prediction, (1x3xHxW) with 1=batch dimension (only batch_size=1 supported atm), 3=scene coordainte dimensions, H=height and W=width.
 * @param neuralGuidanceSrc Neural guidance prediction. Same size as scene coordinate prediction, but only 1-dim instead of 3-dim per pixel (one sampling weight per pixel)
 * @param outPoseSrc Camera pose (output parameter), (4x4) tensor containing the homogeneous camera tranformation matrix.
 * @param ransacHypotheses Number of RANSAC iterations.
 * @param inlierThreshold Inlier threshold for RANSAC in px.
 * @param focalLength Focal length of the camera in px.
 * @param ppointX Coordinate (X) of the prinicpal points.
 * @param ppointY Coordinate (Y) of the prinicpal points.
 * @param inlierAlpha Alpha parameter for soft inlier counting.
 * @param inlierBeta Beta parameter for soft inlier counting.
 * @param maxReproj Reprojection errors are clamped above this value (px).
 * @param subSampling Sub-sampling  of the scene coordinate prediction wrt the input image.
 */
void ngdsac_forward(
	at::Tensor sceneCoordinatesSrc, 
	at::Tensor neuralGuidanceSrc,
	at::Tensor outPoseSrc,
	int ransacHypotheses, 
	float inlierThreshold,
	float focalLength,
	float ppointX,
	float ppointY,
	float inlierAlpha,
	float inlierBeta,
	float maxReproj,
	int subSampling)
{
	ThreadRand::init();

	// access to tensor objects
	ngdsac::coord_t sceneCoordinates = 
		sceneCoordinatesSrc.accessor<float, 4>();

	ngdsac::ng_t neuralGuidance = 
		neuralGuidanceSrc.accessor<float, 4>();

	// dimensions of scene coordinate predictions
	int imH = sceneCoordinates.size(2);
	int imW = sceneCoordinates.size(3);

	// internal camera calibration matrix
	cv::Mat_<float> camMat = cv::Mat_<float>::eye(3, 3);
	camMat(0, 0) = focalLength;
	camMat(1, 1) = focalLength;
	camMat(0, 2) = ppointX;
	camMat(1, 2) = ppointY;	

	// calculate original image position for each scene coordinate prediction
	cv::Mat_<cv::Point2i> sampling = 
		ngdsac::createSampling(imW, imH, subSampling, 0, 0);

	std::cout << BLUETEXT("Sampling " << ransacHypotheses << " hypotheses.") << std::endl;
	StopWatch stopW;

	// sample RANSAC hypotheses
	std::vector<ngdsac::pose_t> hypotheses;
	std::vector<std::vector<cv::Point2i>> sampledPoints;  
	std::vector<std::vector<cv::Point2f>> imgPts;
	std::vector<std::vector<cv::Point3f>> objPts;

	ngdsac::sampleHypotheses(
		sceneCoordinates,
		neuralGuidance,
		sampling,
		camMat,
		ransacHypotheses,
		inlierThreshold,
		hypotheses,
		sampledPoints,
		imgPts,
		objPts);

	std::cout << "Done in " << stopW.stop() / 1000 << "s." << std::endl;	
	std::cout << BLUETEXT("Calculating scores.") << std::endl;
    
	// compute reprojection error images
	std::vector<cv::Mat_<float>> reproErrs(ransacHypotheses);
	cv::Mat_<double> jacobeanDummy;

	#pragma omp parallel for 
	for(unsigned h = 0; h < hypotheses.size(); h++)
    	reproErrs[h] = ngdsac::getReproErrs(
		sceneCoordinates,
		hypotheses[h], 
		sampling, 
		camMat,
		maxReproj,
		jacobeanDummy);

    // soft inlier counting
	std::vector<double> scores = ngdsac::getHypScores(
    	reproErrs,
    	inlierThreshold,
    	inlierAlpha,
    	inlierBeta);

	std::cout << "Done in " << stopW.stop() / 1000 << "s." << std::endl;
	std::cout << BLUETEXT("Drawing final hypothesis.") << std::endl;	

	// apply soft max to scores to get a distribution
	std::vector<double> hypProbs = ngdsac::softMax(scores);
	double hypEntropy = ngdsac::entropy(hypProbs); // measure distribution entropy
	int hypIdx = ngdsac::draw(hypProbs, false); // select winning hypothesis

	std::cout << "Soft inlier count: " << scores[hypIdx] << " (Selection Probability: " << (int) (hypProbs[hypIdx]*100) << "%)" << std::endl; 
	std::cout << "Entropy of hypothesis distribution: " << hypEntropy << std::endl;


	std::cout << "Done in " << stopW.stop() / 1000 << "s." << std::endl;
	std::cout << BLUETEXT("Refining winning pose:") << std::endl;

	// refine selected hypothesis
	cv::Mat_<int> inlierMap;

	ngdsac::refineHyp(
		sceneCoordinates,
		reproErrs[hypIdx],
		sampling,
		camMat,
		inlierThreshold,
		MAX_REF_STEPS,
		maxReproj,
		hypotheses[hypIdx],
		inlierMap);

	std::cout << "Done in " << stopW.stop() / 1000 << "s." << std::endl;

	// write result back to PyTorch
	ngdsac::trans_t estTrans = ngdsac::pose2trans(hypotheses[hypIdx]);

	auto outPose = outPoseSrc.accessor<float, 2>();
	for(unsigned x = 0; x < 4; x++)
	for(unsigned y = 0; y < 4; y++)
		outPose[y][x] = estTrans(y, x);	
}

/**
 * @brief Performs pose estimation, and calculates the gradients of the pose loss wrt to scene coordinates and neural guidance.
 * @param sceneCoordinatesSrc Scene coordinate prediction, (1x3xHxW) with 1=batch dimension (only batch_size=1 supported atm), 3=scene coordainte dimensions, H=height and W=width.
 * @param outSceneCoordinatesGradSrc Scene coordinate gradients (output parameter). (1x3xHxW) same as scene coordinate input.
 * @param neuralGuidanceSrc Neural guidance prediction. Same size as scene coordinate prediction, but only 1-dim instead of 3-dim per pixel (one sampling weight per pixel)
 * @param outNeuralGuidanceGradSrc Neural guidance gradients (output parameter). Same size as neural guidance prediction.
 * @param gtPoseSrc Ground truth camera pose, (4x4) tensor.
 * @param shiftX Horizontal offset in px in case the input image has been shifted before scene coordinate predictions.
 * @param shiftY Vertical offset in px in case the input image has been shifted before scene coordinate predictions.
 * @param ransacHypotheses Number of RANSAC iterations.
 * @param inlierThreshold Inlier threshold for RANSAC in px.
 * @param focalLength Focal length of the camera in px.
 * @param ppointX Coordinate (X) of the prinicpal points.
 * @param ppointY Coordinate (Y) of the prinicpal points.
 * @param wLossRot Weight of the rotation loss term.
 * @param wLossTrans Weight of the translation loss term.
 * @param inlierAlpha Alpha parameter for soft inlier counting.
 * @param inlierBeta Beta parameter for soft inlier counting.
 * @param maxReproj Reprojection errors are clamped above this value (px).
 * @param subSampling Sub-sampling  of the scene coordinate prediction wrt the input image.
 * @param randomSeed External random seed to make sure we draw different samples across calls of this function.
 * @return DSAC expectation of the pose loss.
 */

double ngdsac_backward(
	at::Tensor sceneCoordinatesSrc, 
	at::Tensor outSceneCoordinatesGradSrc, 
	at::Tensor neuralGuidanceSrc,
	at::Tensor outNeuralGuidanceGradSrc,
	at::Tensor gtPoseSrc, 
	int shiftX, 
	int shiftY,
	int ransacHypotheses,
	float inlierThreshold,
	float focalLength,
	float ppointX,
	float ppointY,
	float wLossRot,
	float wLossTrans,
	float inlierAlpha,
	float inlierBeta,
	float maxReproj,
	int subSampling,
	int randomSeed)
{
	ThreadRand::init(randomSeed);

	// access to tensor objects
	ngdsac::coord_t sceneCoordinates = 
		sceneCoordinatesSrc.accessor<float, 4>();

	ngdsac::coord_t sceneCoordinatesGrads = 
		outSceneCoordinatesGradSrc.accessor<float, 4>();

	ngdsac::ng_t neuralGuidance = 
		neuralGuidanceSrc.accessor<float, 4>();

	ngdsac::ng_t neuralGuidanceGrads = 
		outNeuralGuidanceGradSrc.accessor<float, 4>();

	// dimensions of scene coordinate predictions
	int imH = sceneCoordinates.size(2);
	int imW = sceneCoordinates.size(3);

	// internal camera calibration matrix
	cv::Mat_<float> camMat = cv::Mat_<float>::eye(3, 3);
	camMat(0, 0) = focalLength;
	camMat(1, 1) = focalLength;
	camMat(0, 2) = ppointX;
	camMat(1, 2) = ppointY;	

	//convert ground truth pose type
	ngdsac::trans_t gtTrans(4, 4);
	auto gtPose = gtPoseSrc.accessor<float, 2>();

	for(unsigned x = 0; x < 4; x++)
	for(unsigned y = 0; y < 4; y++)
		gtTrans(y, x) = gtPose[y][x];

	// calculate original image position for each scene coordinate prediction
	cv::Mat_<cv::Point2i> sampling = 
		ngdsac::createSampling(imW, imH, subSampling, shiftX, shiftY);

	// sample RANSAC hypotheses
	std::cout << BLUETEXT("Sampling " << ransacHypotheses << " hypotheses.") << std::endl;
	StopWatch stopW;

	std::vector<ngdsac::pose_t> initHyps;
	std::vector<std::vector<cv::Point2i>> sampledPoints;  
	std::vector<std::vector<cv::Point2f>> imgPts;
	std::vector<std::vector<cv::Point3f>> objPts;

	ngdsac::sampleHypotheses(
		sceneCoordinates,
		neuralGuidance,
		sampling,
		camMat,
		ransacHypotheses,
		inlierThreshold,
		initHyps,
		sampledPoints,
		imgPts,
		objPts);

	// gradients of (log) neural guidance arise from an indicator tensor 
	// where we mark which minimal sets have been sampled
	for(unsigned h = 0; h < initHyps.size(); h++)
	for(int j = 0; j < 4; j++)
	{
		int x = sampledPoints[h][j].x;
		int y = sampledPoints[h][j].y;

		neuralGuidanceGrads[0][0][y][x] += 1;
	}

	std::cout << "Done in " << stopW.stop() / 1000 << "s." << std::endl;	
    std::cout << BLUETEXT("Calculating scores.") << std::endl;

	// compute reprojection error images
	std::vector<cv::Mat_<float>> reproErrs(ransacHypotheses);
	std::vector<cv::Mat_<double>> jacobeansHyp(ransacHypotheses);

	#pragma omp parallel for 
	for(unsigned h = 0; h < initHyps.size(); h++)
    	reproErrs[h] = ngdsac::getReproErrs(
		sceneCoordinates,
		initHyps[h], 
		sampling, 
		camMat,
		maxReproj,
		jacobeansHyp[h],
		true);

    // soft inlier counting
	std::vector<double> scores = ngdsac::getHypScores(
    	reproErrs,
    	inlierThreshold,
    	inlierAlpha,
    	inlierBeta);

	std::cout << "Done in " << stopW.stop() / 1000 << "s." << std::endl;
	std::cout << BLUETEXT("Drawing final hypothesis.") << std::endl;	

	// apply soft max to scores to get a distribution
	std::vector<double> hypProbs = ngdsac::softMax(scores);
	double hypEntropy = ngdsac::entropy(hypProbs); // measure distribution entropy
	std::cout << "Entropy: " << hypEntropy << std::endl;

	std::cout << "Done in " << stopW.stop() / 1000 << "s." << std::endl;
	std::cout << BLUETEXT("Refining poses:") << std::endl;

	// collect inliers and refine poses
	std::vector<ngdsac::pose_t> refHyps(ransacHypotheses);
	std::vector<cv::Mat_<int>> inlierMaps(refHyps.size());
	
	#pragma omp parallel for
	for(unsigned h = 0; h < refHyps.size(); h++)
	{
		refHyps[h].first = initHyps[h].first.clone();
		refHyps[h].second = initHyps[h].second.clone();

		if(hypProbs[h] < PROB_THRESH) continue;

		ngdsac::refineHyp(
			sceneCoordinates,
			reproErrs[h],
			sampling,
			camMat,
			inlierThreshold,
			MAX_REF_STEPS,
			maxReproj,
			refHyps[h],
			inlierMaps[h]);
	}

	std::cout << "Done in " << stopW.stop() / 1000 << "s." << std::endl;
	
	// calculate expected pose loss
	double expectedLoss = 0;
	double lossSoftClamp = 100000; // high value pratically disables soft clamping
	std::vector<double> losses(refHyps.size());

	for(unsigned h = 0; h < refHyps.size(); h++)
	{
		ngdsac::trans_t estTrans = ngdsac::pose2trans(refHyps[h]);
		losses[h] = ngdsac::loss(estTrans, gtTrans, wLossRot, wLossTrans, lossSoftClamp);
		expectedLoss += hypProbs[h] * losses[h];
	}
	
   	// === doing the backward pass ====================================================================
	
	// acumulate hypotheses gradients for patches
	cv::Mat_<double> dLoss_dObj = cv::Mat_<double>::zeros(sampling.rows * sampling.cols, 3);

    // --- path I, hypothesis path --------------------------------------------------------------------
    std::cout << BLUETEXT("Calculating gradients wrt hypotheses.") << std::endl;

    // precalculate gradients per of hypotheis wrt object coordinates
    std::vector<cv::Mat_<double>> dHyp_dObjs(refHyps.size());

    #pragma omp parallel for
    for(unsigned h = 0; h < refHyps.size(); h++)
    {
		int batchIdx = 0; // only batch size=1 supported atm

        // differentiate refinement around optimum found in last optimization iteration
        dHyp_dObjs[h] = cv::Mat_<double>::zeros(6, sampling.rows * sampling.cols * 3);

        if(hypProbs[h] < PROB_THRESH) continue; // skip hypothesis with no impact on expectation

        // collect inlier correspondences of last refinement iteration
        std::vector<cv::Point2f> imgPts;
        std::vector<cv::Point2i> srcPts;
        std::vector<cv::Point3f> objPts;

        for(int x = 0; x < inlierMaps[h].cols; x++)
        for(int y = 0; y < inlierMaps[h].rows; y++)
        {
            if(inlierMaps[h](y, x))
            {
                imgPts.push_back(sampling(y, x));
                srcPts.push_back(cv::Point2i(x, y));
                objPts.push_back(cv::Point3f(
					sceneCoordinates[batchIdx][0][y][x],
					sceneCoordinates[batchIdx][1][y][x],
					sceneCoordinates[batchIdx][2][y][x]));
            }
        }

        if(imgPts.size() < 4)
            continue;

        // calculate reprojection errors
        std::vector<cv::Point2f> projections;
        cv::Mat_<double> projectionsJ;
        cv::projectPoints(objPts, refHyps[h].first, refHyps[h].second, camMat, cv::Mat(), projections, projectionsJ);

        projectionsJ = projectionsJ.colRange(0, 6);

        //assemble the jacobean of the refinement residuals
        cv::Mat_<double> jacobeanR = cv::Mat_<double> ::zeros(objPts.size(), 6);
        cv::Mat_<double> dNdP(1, 2);
        cv::Mat_<double> dNdH(1, 6);

        for(unsigned ptIdx = 0; ptIdx < objPts.size(); ptIdx++)
        {
            double err = std::max(cv::norm(projections[ptIdx] - imgPts[ptIdx]), EPS);
            if(err > maxReproj)
                continue;

            // derivative of norm
            dNdP(0, 0) = 1 / err * (projections[ptIdx].x - imgPts[ptIdx].x);
            dNdP(0, 1) = 1 / err * (projections[ptIdx].y - imgPts[ptIdx].y);

            dNdH = dNdP * projectionsJ.rowRange(2 * ptIdx, 2 * ptIdx + 2);
            dNdH.copyTo(jacobeanR.row(ptIdx));
        }

        //calculate the pseudo inverse
        jacobeanR = - (jacobeanR.t() * jacobeanR).inv(cv::DECOMP_SVD) * jacobeanR.t();

        double maxJR = ngdsac::getMax(jacobeanR);
        if(maxJR > 10) jacobeanR = 0; // clamping for stability

        cv::Mat rot;
        cv::Rodrigues(refHyps[h].first, rot);

        for(unsigned ptIdx = 0; ptIdx < objPts.size(); ptIdx++)
        {
            cv::Mat_<double> dNdO = ngdsac::dProjectdObj(imgPts[ptIdx], objPts[ptIdx], rot, refHyps[h].second, camMat, maxReproj);
            dNdO = jacobeanR.col(ptIdx) * dNdO;

            int dIdx = srcPts[ptIdx].y * sampling.cols * 3 + srcPts[ptIdx].x * 3;
            dNdO.copyTo(dHyp_dObjs[h].colRange(dIdx, dIdx + 3));
        }
    }

    // combine gradients per hypothesis
    std::vector<cv::Mat_<double>> gradients(refHyps.size());
    ngdsac::pose_t hypGT = ngdsac::trans2pose(gtTrans);

    #pragma omp parallel for
    for(unsigned h = 0; h < refHyps.size(); h++)
    {
		if(hypProbs[h] < PROB_THRESH) continue;

        cv::Mat_<double> dLoss_dHyp = ngdsac::dLoss(refHyps[h], hypGT, wLossRot, wLossTrans, lossSoftClamp);
        gradients[h] = dLoss_dHyp * dHyp_dObjs[h];
    }

	std::cout << "Done in " << stopW.stop() / 1000 << "s." << std::endl;
    
    // --- path II, score path --------------------------------------------------------------------

    std::cout << BLUETEXT("Calculating gradients wrt scores.") << std::endl;

    std::vector<cv::Mat_<double>> dLoss_dScore_dObjs = ngdsac::dSMScore(
    	sceneCoordinates,
    	sampling, 
    	sampledPoints, 
    	losses, 
    	hypProbs, 
    	initHyps, 
    	reproErrs, 
    	jacobeansHyp,
    	camMat,
    	inlierAlpha,
    	inlierBeta,
    	inlierThreshold,
    	maxReproj);

    std::cout << "Done in " << stopW.stop() / 1000 << "s." << std::endl;

    // assemble full gradient tensor
    for(unsigned h = 0; h < refHyps.size(); h++)
    {
		if(hypProbs[h] < PROB_THRESH) continue;
		int batchIdx = 0; // only batch size=1 supported atm

	    for(int idx = 0; idx < sampling.rows * sampling.cols; idx++)
	    {
	    	int x = idx % sampling.cols;
	    	int y = idx / sampling.cols;
    	
	        sceneCoordinatesGrads[batchIdx][0][y][x] += 
	        	hypProbs[h] * gradients[h](idx * 3 + 0) + dLoss_dScore_dObjs[h](idx, 0);
	        sceneCoordinatesGrads[batchIdx][1][y][x] += 
	        	hypProbs[h] * gradients[h](idx * 3 + 1) + dLoss_dScore_dObjs[h](idx, 1);
	        sceneCoordinatesGrads[batchIdx][2][y][x] += 
	        	hypProbs[h] * gradients[h](idx * 3 + 2) + dLoss_dScore_dObjs[h](idx, 2);
	    }
	}

	return expectedLoss;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	m.def("forward", &ngdsac_forward, "NG-DSAC++ forward");
	m.def("backward", &ngdsac_backward, "NG-DSAC++ backward");
}
