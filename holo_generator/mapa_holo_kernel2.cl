// kernel function of openCL program to find the nearest 2D point to a given point
//  m is the number of possible points
//  Xholo is the array for the X coordinate of the desired point
//  Yholo is the array for the Y coordinate of the desired point
//  Xvalues is the array for the X coordinate of the possible points
//  Yvalues is the array for the Y coordinate of the possible points
//  holo is the array for the index of the nearest point

__kernel void nearest(ushort m,
                      __global float *Xholo,
                      __global float *Yholo,
                      __global float *Xvalues,
                      __global float *Yvalues,
                      __global int *holo)
{
    int i = get_global_id(0);

    float minDist = 100;
    float dist;

    for (int j = 0; j<m; j++)
    {
        dist = sqrt(pow(Xholo[i] - Xvalues[j], 2) + pow(Yholo[i] - Yvalues[j], 2));
        if (dist < minDist)
        {
            minDist = dist;
            holo[i] = j;
        }
    }

}





