#include <vector>
#include <fstream>
#include <algorithm>
#include <iostream>

#include "ipoint.h"
#include "limits.h"
#include "float.h"
#include "stdlib.h"
#include "string.h"

#include <sys/types.h>
#include <sys/time.h>
#include <sys/timeb.h>

using namespace std;

extern float get_matches_gpu(IpVec &ipts1, IpVec &ipts2, int &pairs);
extern float get_matches_gpu_new(IpVec &orig, vector<IpVec> &top_vec, vector<int> &top_index, Matches &matches);

#define GET_TIME(start, end, duration)                                     \
   duration.tv_sec = (end.tv_sec - start.tv_sec);                         \
   if (end.tv_usec >= start.tv_usec) {                                     \
      duration.tv_usec = (end.tv_usec - start.tv_usec);                   \
   }                                                                       \
   else {                                                                  \
      duration.tv_usec = (1000000L - (start.tv_usec - end.tv_usec));   \
      duration.tv_sec--;                                                   \
   }                                                                       \
   if (duration.tv_usec >= 1000000L) {                                  \
      duration.tv_sec++;                                                   \
      duration.tv_usec -= 1000000L;                                     \
   }
   

#define RATIO 0.3f
#define THRESHOLD 5

//! Populate IpPairVec with matched ipts 
void get_matches(IpVec &ipts1, IpVec &ipts2, int &pairs)
{
    float dist, d1, d2;
    pairs = 0;

    for(unsigned int i = 0; i < ipts1.size(); i++) 
    {

        if (i >= ipts1.size() * RATIO) {
            if (pairs < THRESHOLD) return;
        }

        d1 = d2 = FLT_MAX;

        for(unsigned int j = 0; j < ipts2.size(); j++) 
        {
            dist = ipts1[i] - ipts2[j];  

            if(dist<d1) // if this feature matches better than current best
            {
                d2 = d1;
                d1 = dist;
            }
            else if(dist<d2) // this feature matches better than second best
            {
                d2 = dist;
            }
        }

        // If match has a d1:d2 ratio < 0.65 ipoints are a match
        if(d1/d2 < 0.65) 
        { 
            // Store the change in position
            pairs++;
        }
    }
}

void get_vector(char* path, IpVec &vec) {
    char img_path[PATH_MAX];
    int dim, len; 
    std::ifstream file(path);

    file >> img_path;
    file >> dim;
    file >> len;

    for (int i = 0; i < len; i++) {
        float temp;
        Ipoint p;

        p.dim = dim;
        file >> p.x;
        file >> p.y;
        file >> p.scale;
        file >> temp;

        for (int j = 0; j < dim; j++) {
            file >> p.descriptor[j];
        }

        vec.push_back(p);
    }

    file.close();
}

bool compare(const ImageMatch &match1, const ImageMatch &match2) {
    return match1.pairs > match2.pairs;
}

int main(int args, char** argv) {

    if (args < 6) {
        printf("USAGE: IMG_NUM TOP_NUM RESULT_PATH POINT_PATH MODE\n");
        exit(-1);
    }

    const int img_num = atoi(argv[1]);
    const int top_num = atoi(argv[2]);
    const char* result_path = argv[3];
    const char* point_path = argv[4];
    const char* mode = argv[5];

    struct timeval  bd_tick_x, bd_tick_e, bd_tick_d;
    gettimeofday(&bd_tick_x, 0);

    // Read the feature vector of the provided image
    IpVec orig;
    char vec_path[PATH_MAX];
    sprintf(vec_path, "%s/%d.txt", point_path, img_num);
    get_vector(vec_path, orig);

    // Read voctree resutls and match with the provided image
    char list_path[PATH_MAX];
    Matches matches;
    sprintf(list_path, "%s/%d.txt", result_path, img_num);
    std::ifstream result_file(list_path);
    float compute_time = 0;
    float gpu_time = 0;

    int skip = 1;
    vector<IpVec> top_vec; 
    vector<int> top_index;

    for (int i = 0; i < top_num; i++) {
        int index;
        if (!(result_file >> index)) break;
        if (skip-- > 0) continue; 

        IpVec vec;
        char path[PATH_MAX];
        sprintf(path, "%s/%d.txt", point_path, index);
        get_vector(path, vec);

        top_vec.push_back(vec);
        top_index.push_back(index);
    }

    result_file.close();

    if (strcmp(mode, "cpu") == 0 || strcmp(mode, "gpu") == 0) {

        for (int i = 0; i < top_num - 1; i++) {

            int pairs;
            IpVec vec = top_vec[i];
            int index = top_index[i];

            struct timeval  tick_x, tick_e, tick_d;
            gettimeofday(&tick_x, 0);

            if (strcmp(mode, "gpu") == 0) {
                gpu_time += get_matches_gpu(orig, vec, pairs);
            } else {
                get_matches(orig, vec, pairs);
            }

            gettimeofday(&tick_e, 0);
            GET_TIME(tick_x, tick_e, tick_d);

            compute_time += tick_d.tv_sec + tick_d.tv_usec/1000000.0f;

            ImageMatch match;
            match.index = index;
            match.pairs = pairs;
            matches.push_back(match);
        }

    } else {
        gpu_time += get_matches_gpu_new(orig, top_vec, top_index, matches);
    }

    // Sort all matches
    std::sort(matches.begin(), matches.end(), compare);

    for (int i = 0; i < 3; i++) {
        std::cout << matches[i].index << std::endl;
    }

    gettimeofday(&bd_tick_e, 0);
    GET_TIME(bd_tick_x, bd_tick_e, bd_tick_d);

    float all_time = bd_tick_d.tv_sec + bd_tick_d.tv_usec/1000000.0f;

    std::cout << gpu_time / 1000.0f << "," << compute_time << "," << all_time - compute_time << std::endl;

}
