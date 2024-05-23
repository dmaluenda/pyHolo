__kernel void nearest(ushort m,  // total number of accecible values 
                      __global float* Xholo,    // Desired real values
                      __global float* Yholo,    // Desired imag values
                      __global float* Xvalues,  // Access. real values
                      __global float* Yvalues,  // Access. imag values
                      __global float* holo )  // Index of nearest point: RESULT to return
{
    int idx = get_global_id(0);  // pixel index under evaluation

    // int idx_debug = -1;  // just for debugging

    float Dx;
    float Dy;
    float best_dist = 100;  // large value just to initialize
    float dist;
    int best_i;

    float desX = Xholo[idx];  // desired real value of current pixel
    float desY = Yholo[idx];  // desired imag value of current pixel

    // if(idx == idx_debug)
    //     printf(" --- %d ---\n" , idx_debug);

    for(int i=0; i<m; i++)
    {
        Dx = desX - Xvalues[i];
        Dy = desY - Yvalues[i];
        
        dist = Dx*Dx + Dy*Dy;

        if(dist < best_dist)
        {
            // if(idx == idx_debug)
            //     printf("Desired: %f,%f ; current: %f,%f ; dist = %f (best_dist = %f)  ; %d <-- IN\n",
            //        desX, desY, Xvalues[i], Yvalues[i], dist, best_dist, i);
            best_dist = dist;
            best_i = i;
        }
        // else{
        //     if(idx == idx_debug)
        //         printf("Desired: %f,%f ; current: %f,%f ; dist = %f (best_dist = %f) ; %d\n",
        //            desX, desY, Xvalues[i], Yvalues[i], dist, best_dist, i);
        // }

    }
    
    holo[idx] = best_i;  // storing the result into the global holo array
}







