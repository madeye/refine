#include <vector>
#include <fstream>
#include <algorithm>
#include <iostream>

#include "ipoint.h"
#include "limits.h"
#include "float.h"
#include "stdlib.h"

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

    if (args < 5) {
        printf("USAGE: IMG_NUM TOP_NUM RESULT_PATH POINT_PATH\n");
        exit(-1);
    }

    const int img_num = atoi(argv[1]);
    const int top_num = atoi(argv[2]);
    const char* result_path = argv[3];
    const char* point_path = argv[4];

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
    int skip = 1;
    for (int i = 0; i < top_num; i++) {
        int index;
        if (!(result_file >> index)) break;
        if (skip-- > 0) continue; 

        IpVec vec;
        char path[PATH_MAX];
        sprintf(path, "%s/%d.txt", point_path, index);
        get_vector(path, vec);

        int pairs;
        get_matches(orig, vec, pairs);

        ImageMatch match;
        match.index = index;
        match.pairs = pairs;
        matches.push_back(match);
    }
    result_file.close();

    // Sort all matches
    std::sort(matches.begin(), matches.end(), compare);

    for (int i = 0; i < 3; i++) {
        std::cout << matches[i].index << std::endl;
    }

}
