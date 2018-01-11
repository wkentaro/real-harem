<img src="https://drive.google.com/uc?id=174cjyuSrj47D5s_xmSQnbSmXM75WcvHG" align="right" width="300" />

# REAL HAREM

*Transgender of real persons to achieve real harem, with mixed reality on Hololens.*

This is a project in
[the lecture "Intelligent Software"](http://www.mi.t.u-tokyo.ac.jp/ushiku/lectures/is/)
at the University of Tokyo in winter 2017.

<img src="https://drive.google.com/uc?id=1TiDzASgw_E70PSEJIX5rvML-uQ83a9_8" />


## Challenges

- [x] Translate gender of person with remaining his/her face characteristic.
- [x] Track face motion in real time.
- [x] Paste the generated image onto the tracked face naturally. (transgender network won't run in real time, 60fps）
- [ ] Translate gender of clothes.


## Features


### Convert faces to female

Author: [@wkentaro](https://github.com/wkentaro)

**Sample**

```bash
make sample_transgender  # it translates gender of JSK lab members.
```

![](.readme/stargan_transgender_jsk.jpg)


### Paste face to face

Author: [@wkentaro](https://github.com/wkentaro)

**Sample**

```bash
make sample_paste_face  # it paste face to face.
```

![](.readme/face2face_wkentaro.jpg)


## Shared data

Google Drive: https://drive.google.com/open?id=1H8EkgFOWPfjuBbdn_w3cQhvdkZgLp7vo
