/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <algorithm>
#include <iostream>
#include <iterator>
#include <math.h>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::string;
using std::vector;

void ParticleFilter::init(double x, double y, double theta, double std[])
{
  /**
   * TODO: Set the number of particles. Initialize all particles to
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1.
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method
   *   (and others in this file).
   */
  std::default_random_engine gen;
  num_particles = 10;
  std::normal_distribution<double> dist_x(x, std[0]);
  std::normal_distribution<double> dist_y(y, std[1]);
  std::normal_distribution<double> dist_theta(theta, std[2]);

  for (unsigned int i = 0; i < num_particles; i++)
  {
    Particle temp;
    temp.id = i;
    temp.x = dist_x(gen);
    temp.y = dist_y(gen);
    temp.theta = dist_theta(gen);
    temp.weight = 1.0;

    particles.push_back(temp);
    weights.push_back(temp.weight);
  }
  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[],
                                double velocity, double yaw_rate)
{
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */
  std::normal_distribution<double> N_dis_x(0, std_pos[0]);
  std::normal_distribution<double> N_dis_y(0, std_pos[1]);
  std::normal_distribution<double> N_dis_theta(0, std_pos[2]);
  std::default_random_engine gen;

  for (int i = 0; i < num_particles; i++)
  {
    if (fabs(yaw_rate) < 0.0001)
    {
      particles[i].x += velocity * delta_t * cos(particles[i].theta);
      particles[i].y += velocity * delta_t * sin(particles[i].theta);
    }
    else
    {
      particles[i].x += velocity / yaw_rate * (sin(particles[i].theta + yaw_rate * delta_t) - sin(particles[i].theta));
      particles[i].y += velocity / yaw_rate * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate * delta_t));
      particles[i].theta += yaw_rate * delta_t;
    }

    particles[i].x += N_dis_x(gen);
    particles[i].y += N_dis_y(gen);
    particles[i].theta += N_dis_theta(gen);
  }
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted,
                                     vector<LandmarkObs> &observations)
{
  /**
   * TODO: Find the predicted measurement that is closest to each
   *   observed measurement and assign the observed measurement to this
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will
   *   probably find it useful to implement this method and use it as a helper
   *   during the updateWeights phase.
   */
  for (auto &obs : observations)
  {
    double min_dist = std::numeric_limits<double>::max();
    int map_id = -1;
    for (auto &pred : predicted)
    {
      double cur_distance = dist(obs.x, obs.y, pred.x, pred.y);
      if (cur_distance < min_dist)
      {
        min_dist = cur_distance;
        map_id = pred.id;
      }
    }
    obs.id = map_id;
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   const vector<LandmarkObs> &observations,
                                   const Map &map_landmarks)
{
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian
   *   distribution. You can read more about this distribution here:
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system.
   *   Your particles are located according to the MAP'S coordinate system.
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */
  for (unsigned int i = 0; i < num_particles; i++)
  {
    double p_x = particles[i].x;
    double p_y = particles[i].y;
    double p_theta = particles[i].theta;

    vector<LandmarkObs> predictions;
    for (auto lm : map_landmarks.landmark_list)
    {
      if ((fabs(lm.x_f - p_x) <= sensor_range) && (fabs(lm.y_f - p_y) <= sensor_range))
      {
        predictions.push_back(LandmarkObs{lm.id_i, lm.x_f, lm.y_f});
      }
    }

    vector<LandmarkObs> transformed_os;
    for (auto obs : observations)
    {
      double t_x = cos(p_theta) * obs.x - sin(p_theta) * obs.y + p_x;
      double t_y = sin(p_theta) * obs.x + cos(p_theta) * obs.y + p_y;
      transformed_os.push_back(LandmarkObs{obs.id, t_x, t_y});
    }

    dataAssociation(predictions, transformed_os);

    particles[i].weight = 1.0;

    for (auto tr : transformed_os)
    {
      double pr_x, pr_y;
      std::for_each(predictions.begin(), predictions.end(),
                    [&](LandmarkObs lm) {
                      if (lm.id == tr.id)
                      {
                        pr_x = lm.x;
                        pr_y = lm.y;
                      }
                    });

      double s_x = std_landmark[0];
      double s_y = std_landmark[1];
      double obs_w = (1 / (2 * M_PI * s_x * s_y)) * exp(-(pow(pr_x - tr.x, 2) / (2 * pow(s_x, 2)) + (pow(pr_y - tr.y, 2) / (2 * pow(s_y, 2)))));

      particles[i].weight *= obs_w;
    }
  }
}

void ParticleFilter::resample()
{
  /**
   * TODO: Resample particles with replacement with probability proportional
   *   to their weight.
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
  vector<Particle> new_particles;
  std::default_random_engine gen;
  // get all of the current weights
  vector<double> weights;
  for (int i = 0; i < num_particles; i++)
  {
    weights.push_back(particles[i].weight);
  }

  // generate random starting index for resampling wheel
  std::uniform_int_distribution<int> uniintdist(0, num_particles - 1);
  auto index = uniintdist(gen);

  // get max weight
  double max_weight = *max_element(weights.begin(), weights.end());

  // uniform random distribution [0.0, max_weight)
  std::uniform_real_distribution<double> unirealdist(0.0, max_weight);

  double beta = 0.0;

  // spin the resample wheel!
  for (int i = 0; i < num_particles; i++)
  {
    beta += unirealdist(gen) * 2.0;
    while (beta > weights[index])
    {
      beta -= weights[index];
      index = (index + 1) % num_particles;
    }
    new_particles.push_back(particles[index]);
  }

  particles = new_particles;
}

void ParticleFilter::SetAssociations(Particle &particle,
                                     const vector<int> &associations,
                                     const vector<double> &sense_x,
                                     const vector<double> &sense_y)
{
  // particle: the particle to which assign each listed association,
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations = associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1); // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord)
{
  vector<double> v;

  if (coord == "X")
  {
    v = best.sense_x;
  }
  else
  {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1); // get rid of the trailing space
  return s;
}