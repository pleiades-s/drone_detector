import os





Drlist=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]

for dr in Drlist:
    try:
        os.makedirs(os.path.join(str(dr)))
    except OSError as e:
        print("fail")

