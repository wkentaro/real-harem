# REAL HAREM

*Transgender of real persons to achieve real harem, with mixed reality on Hololens.*

This is a project in
[the lecture "Intelligent Software"](http://www.mi.t.u-tokyo.ac.jp/ushiku/lectures/is/)
at the University of Tokyo in winter 2017.

<img src="https://drive.google.com/uc?id=1TiDzASgw_E70PSEJIX5rvML-uQ83a9_8" width="50%" /> <img src="https://drive.google.com/uc?id=174cjyuSrj47D5s_xmSQnbSmXM75WcvHG" width="36.5%" />


## Challenges

- [x] Translate gender of person with remaining his/her face characteristic.
- [x] Track face motion in real time.
- [x] Paste the generated image onto the tracked face naturally. (transgender network won't run in real time, 60fps）
- [ ] Translate gender of clothes.


## Usage


### Hololens demo

**Server**

```bash
# installation
make sample_transgender

cd transgender
sudo bash
source .anaconda3/bin/activate
./runserver.py

# client sample
./sample_post.py
```

**Client (Hololens)**

Author: [@ban-masa](https://github.com/ban-masa)

Deploy https://github.com/ban-masa/HoloFaceSamples/tree/real-harem to Hololens.  
*Please check the URL of server.*


### Convert faces to female

Author: [@wkentaro](https://github.com/wkentaro)

We use [StarGAN](https://github.com/yunjey/StarGAN) for transgender.

```bash
make sample_transgender  # it translates gender of JSK lab members.
```

![](.readme/stargan_transgender_jsk.jpg)


### Paste face to face

Author: [@wkentaro](https://github.com/wkentaro)

```bash
make sample_paste_face  # it paste face to face.
```

![](.readme/face2face_wkentaro.jpg)


## Shared data

- [All data](https://drive.google.com/open?id=1H8EkgFOWPfjuBbdn_w3cQhvdkZgLp7vo)
- [Proposal ppt](https://drive.google.com/open?id=1H8NVqHPfGqQhsGrGBs5R1TFWZF2E-VRV)
- [Rinko (Survey) ppt](https://drive.google.com/open?id=1c0P3DTVWtyem8r3Z6t-XENbdEOkhSJTk)
- [Final demo ppt](https://drive.google.com/open?id=141NIbCRpNpCgqckLlYkeq5hXzsmcRZqx)
