/*********************************************************** 
 *  --- OpenSURF ---                                        *
 *  This library is distributed under the GNU GPL. Please   *
 *  contact chris.evans@irisys.co.uk for more information.  *
 *                                                          *
 *  C. Evans, Research Into Robust Visual Features,         *
 *  MSc University of Bristol, 2008.                        *
 *                                                          *
 ************************************************************/

#ifndef IPOINT_H
#define IPOINT_H

#include <vector>
#include <math.h>

//-------------------------------------------------------

class Ipoint; // Pre-declaration
class ImageMatch;
typedef std::vector<Ipoint> IpVec;
typedef std::vector<ImageMatch> Matches;

#define MAX_DIM 128

//-------------------------------------------------------

//! Ipoint operations
void getMatches(IpVec &ipts1, IpVec &ipts2, int &pairs);

//-------------------------------------------------------

class ImageMatch {
    public:

        ~ImageMatch() {};
        ImageMatch() {};

        int index;
        int pairs;

};

class Ipoint {

    public:

        //! Destructor
        ~Ipoint() {};

        //! Constructor
        Ipoint() {}; 

        //! Dim
        int dim;

        //! Gets the distance in descriptor space between Ipoints
        float operator-(const Ipoint &rhs)
        {
            float sum=0;
            for(int i=0; i < dim; ++i)
                sum += (this->descriptor[i] - rhs.descriptor[i])*(this->descriptor[i] - rhs.descriptor[i]);
            //return sqrt(sum);
            return sum;
        };

        //! Coordinates of the detected interest point
        float x, y;

        //! Detected scale
        float scale;

        //! Vector of descriptor components
        float descriptor[MAX_DIM];

};

//-------------------------------------------------------


#endif
