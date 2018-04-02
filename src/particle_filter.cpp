/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h>
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of
	//   x, y, theta and their uncertainties from GPS) and all weights to 1.
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

	// set the number of particles
	num_particles = 12;

	// create a random number engine
	default_random_engine gen;

	// create normal (Gaussian) distributions for x, y, and theta;
	normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_theta(theta, std[2]);

	cout << "Init" << endl;

	// create the new particles and add them to the particles vector
	for (int i = 0; i < num_particles; i++){
		Particle particle;
		particle.id = i;
		particle.x = dist_x(gen);
		particle.y = dist_y(gen);
		particle.theta = dist_theta(gen);
		particle.weight = 1.0;
		particles.push_back(particle);
		//cout << "Init P" << i << ": x=" << particle.x << " y=" << particle.y << " theta=" << particle.theta << " weight=" << particle.weight << endl;
	}

	// initialize the weights to 1
	for (int i = 0; i < num_particles; i++){
		weights.push_back(1.0);
	}

  // filter is now initialized
	is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	// create a random number engine
	default_random_engine gen;

	//cout << "Prediction (delta_t, velocity, yaw_rate)(" << delta_t << ", " << velocity << ',' << yaw_rate << ")" << endl;

	// create normal (Gaussian) distributions with mean = 0;
	normal_distribution<double> dist_x(0, std_pos[0]);
	normal_distribution<double> dist_y(0, std_pos[1]);
	normal_distribution<double> dist_theta(0, std_pos[2]);

	// update particles vector and check for division by zero if yaw_rate is very small
	for (int i = 0; i < num_particles; i++){
		double pred_x, pred_y, pred_theta;

		if (fabs(yaw_rate) > 0.001) {
			// calculate predictions, lesson 15, # 7 & 8
			pred_x     = particles[i].x + (velocity / yaw_rate) * (sin(particles[i].theta + (yaw_rate * delta_t)) - sin(particles[i].theta));
			pred_y     = particles[i].y + (velocity / yaw_rate) * (cos(particles[i].theta) - cos(particles[i].theta + (yaw_rate * delta_t)));
			pred_theta = particles[i].theta + (yaw_rate * delta_t);
		}
		// don't divide with zero or very small yaw_rate, probably not likely
		else {
			pred_x = particles[i].x + (velocity * delta_t * cos(particles[i].theta));
			pred_y = particles[i].y + (velocity * delta_t * sin(particles[i].theta));
			pred_theta = particles[i].theta;
		}

		// update the particles vector and add gaussian noise
		particles[i].x = pred_x + dist_x(gen);
		particles[i].y = pred_y + dist_y(gen);
		particles[i].theta = pred_theta + dist_theta(gen);

		//cout << "P" << i << " (x, y, tetha, weight):(" << particles[i].x << " ," << particles[i].y << " ," << particles[i].theta << " ," << particles[i].weight << ")" << endl;
	}

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
	//   implement this method and use it as a helper during the updateWeights phase.

	// loop through the observations
	for (int i = 0; i < observations.size(); i++){
		// current observation
		//LandmarkObs curr_obs = observations[i];

		// initialize the minimum distance to a very high number
		double min_dist = 1000.0;

		// initialize map_id to unknown id, e.g. -1
		// if map_id = -1 in updateWeights, then weight is zero for the particle
		int map_id = -1;

		// loop through the predicted vector
		for (int j = 0; j < predicted.size(); j++) {

			// calculate distance
			double curr_dist = dist(observations[i].x, observations[i].y, predicted[j].x, predicted[j].y);

			// check for minimum distance
			if (curr_dist < min_dist) {
				min_dist = curr_dist;
				map_id = predicted[j].id;
			}
		}
		// set the found map_id to the current observation
		observations[i].id = map_id;
		//cout << "dA Predicted map_id" << map_id << " : curr_obs(x,y)= (" << observations[i].x << ", " << observations[i].y << ")" << endl;
	}

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

	// get the standard deviations
	double std_x = std_landmark[0];
	double std_y = std_landmark[1];
	double twopowtwo_std_x = 2 * pow(std_x, 2);
	double twopowtwo_std_y = 2 * pow(std_y, 2);

	// calculate normalization term
	double gauss_norm = 1 / (2 * M_PI * std_x * std_y);

	// loop through all the particles
	for (int i = 0; i < num_particles; i++){
		// get particle coordinates
		double p_x = particles[i].x;
		double p_y = particles[i].y;
		double p_theta = particles[i].theta;
		double cos_p_theta = cos(p_theta);
		double sin_p_theta = sin(p_theta);

		// create vector to hold the predicted landmarks _within_ sensor range
		vector<LandmarkObs> predicted;

		//loop through landmarks
		for (int j = 0; j < map_landmarks.landmark_list.size(); j++) {

			// get landmark coordinates
			int    lm_id = map_landmarks.landmark_list[j].id_i;
			double lm_x  = map_landmarks.landmark_list[j].x_f;
			double lm_y  = map_landmarks.landmark_list[j].y_f;

			// check if the landmark is within the particles sensor range
			double d = dist(p_x, p_y, lm_x, lm_y);
			if (fabs(d) <= sensor_range) {
			// TODO: check if it is faster to use a rectangular boundingbox instead
			// checked it - it doesn't matter. then rather use dist()
			//if (fabs(p_x-lm_x) <= sensor_range && fabs(p_y-lm_y) <= sensor_range) {
				// add the landmark to the list of predicted landmarks
				LandmarkObs lm;
				lm.id = lm_id;
				lm.x = lm_x;
				lm.y = lm_y;
				predicted.push_back(lm);
				//cout << "UW Landmark  " << lm.id << ": x=" << lm.x << " y= " << lm.y << endl;
			}
		}

		// transform the observations from vechicle coords to map coordinates,
		// rotate them to the particle position and save them in a vector
		vector<LandmarkObs> transformed_observations;
		
		// loop through observations
		for (int j = 0; j < observations.size(); j++) {
			// transform and rotate
			double to_x = p_x + ((cos_p_theta * observations[j].x) - (sin_p_theta * observations[j].y));
			double to_y = p_y + ((sin_p_theta * observations[j].x) + (cos_p_theta * observations[j].y));
			LandmarkObs lm;
			lm.id = observations[j].id;
			lm.x = to_x;
			lm.y = to_y;
			transformed_observations.push_back(lm);
			//cout << "UW Obs(x,y)(" << observations[j].x << "," << observations[j].y << ")-->TObs(x,y)(" << lm.x << "," << lm.y << ")" << endl;
		}

		// call dataAssociation() for the current particle to get nearest landmark id
		dataAssociation(predicted, transformed_observations);

		// calculate weight for the current particle
		// reset the weight first
		particles[i].weight = 1.0;

		// clear previous associations vectors
		particles[i].associations.clear();
		particles[i].sense_x.clear();
		particles[i].sense_y.clear();

		// loop through the transformed observations, where .id now is the nearest landmark
		for (int j = 0; j < transformed_observations.size(); j++) {
			// variables to hold coordinates and id
			double to_x  = transformed_observations[j].x;
			double to_y  = transformed_observations[j].y;
			int    to_id = transformed_observations[j].id;

			// if no landmarks were found (.id = -1), set weight to zero
			if (to_id == -1) {
				particles[i].weight = 0.0;
				cout << "UW landmark_id = -1";
			}
			else {
				// add values to associations vectors
				//particles[i].associations.push_back(to_id);
				//particles[i].sense_x.push_back(to_x);
				//particles[i].sense_y.push_back(to_y);

				// find the landmark in predicted vector to get landmark coordinates
				double pred_x, pred_y;
				for (int k = 0; k < predicted.size(); k++) {
					if (predicted[k].id == to_id) {
						pred_x = predicted[k].x;
						pred_y = predicted[k].y;
						//cout << "UW Pred Landmark " << to_id << " : pred(x,y)= (" << pred_x << ", " << pred_y << ") transobs(x,y)= (" << to_x << ", " << to_y << ")" << endl;
					}
				}

				// now use a multivariate Gaussian to the calculate weight for current particle
				// calculate exponent
				double exponent = (pow(pred_x - to_x, 2) / twopowtwo_std_x) + (pow(pred_y - to_y, 2) / twopowtwo_std_y);

				// calculate weight using normalization terms and exponent
				double weight = gauss_norm * exp(-exponent);

				//cout << "UW particle " << i << " weight: " << weight << endl;

				// product the weights
				particles[i].weight *= weight;
			}
			weights[i] = particles[i].weight;
		}

	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight.
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

	// create a random number engine
	default_random_engine gen;

	// create new set of particles
	vector<Particle> resampled_particles;

	// create discrete_distribution using iterators on weights
	discrete_distribution<int> discr_distr(weights.begin(), weights.end());

	// resample with probability proportional to weights
	for (int i = 0; i < num_particles; i++) {
		resampled_particles.push_back(particles[discr_distr(gen)]);
	}

	// Replace the current particles to resampled particles
	particles = resampled_particles;

}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations,
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
