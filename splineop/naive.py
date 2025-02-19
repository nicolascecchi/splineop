import numpy as np 


def naive_bkps(y, K, msl=3):
    """
    y (np.array) : Array of observations
    K (int) : Nb of breakpoints

    """
    x = np.abs(np.diff(y, 3))
    x = np.argsort(x)
    
    x = x+3
    x = x[x<(len(y)-msl)]
    x = x.tolist()
    first_item = int(x.pop())
    bkps = np.array([first_item])
    
    
    while len(bkps) < K:
        new_item = x.pop()

        abs_mindist = np.min(np.abs(new_item-bkps))
        closest_idx = np.argmin(np.abs(new_item-bkps))
        mindist = np.min(new_item-bkps[closest_idx])
        
        
        if abs_mindist > 3:
            if mindist < 0:
                # new_item smaller than the closest bkps
                bkps = np.insert(bkps, closest_idx, new_item)
            else:
                # new_item is bigger than the closest bkp
                closest_idx = closest_idx + 1  
                bkps = np.insert(bkps, closest_idx, new_item)
        else:
            if mindist > 0:
                new_item = np.floor((new_item + bkps[closest_idx])/2) 
            else:
                new_item = np.ceil((new_item + bkps[closest_idx])/2)
            bkps[closest_idx] = new_item
    return bkps.astype(int)
            

def apply_reduce_array(arr, K, msl):
    return naive_bkps(arr, K, msl=msl)

def add_empty(x):
    e = [[]]
    new = e + x
    return new
