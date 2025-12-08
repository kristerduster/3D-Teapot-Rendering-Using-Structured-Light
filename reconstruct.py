import matplotlib.pyplot as plt
import numpy as np
import pickle
from utils import decode, reconstruct


for thresh in [0.001, 0.01, 0.02, 0.03, 0.04]:
    code, mask = decode('images/teapot/grab_6_u/frame_C1_', 20, thresh)
    plt.figure(figsize=(10,4))
    plt.suptitle(f"Threshold = {thresh}")
    
    plt.subplot(1,2,1)
    plt.imshow(code, cmap='jet')
    plt.title("Decoded Code")
    plt.colorbar()
    
    plt.subplot(1,2,2)
    plt.imshow(mask, cmap='gray')
    plt.title("Mask")
    plt.colorbar()
    
    plt.show()

# thresh0C0V = 0.02
# thresh0C0H = 0.02
# thresh0C1V = 0.02
# thresh0C1H = 0.02

# thresh1C0V = 0.02
# thresh1C0H = 0.02
# thresh1C1V = 0.02
# thresh1C1H = 0.02

# thresh2C0V = 0.02
# thresh2C0H = 0.02
# thresh2C1V = 0.02
# thresh2C1H = 0.02

# thresh3C0V = 0.02
# thresh3C0H = 0.02
# thresh3C1V = 0.02
# thresh3C1H = 0.02

# thresh4C0V = 0.02
# thresh4C0H = 0.02
# thresh4C1V = 0.02
# thresh4C1H = 0.02

# thresh5C0V = 0.02
# thresh5C0H = 0.02
# thresh5C1V = 0.02
# thresh5C1H = 0.02

# thresh6C0V = 0.02
# thresh6C0H = 0.02
# thresh6C1V = 0.02
# thresh6C1H = 0.02
