# REAL HAREM


## 背景 / 目的

現実世界における職場や学校には性別の偏りが多く、
異性の目を気にしない習慣や環境への満足度の低下などの弊害を引き起こしていると考えられる。
そこで本プロジェクトでは、
**HMDを使って現実世界にいる人物の性別を転換し、異性がたくさんいる世界を演出する**
ことを目的とする。


## 技術的課題

1. 人の顔を上手くその人の特徴を残したまま性別転換すること。
1. リアルタイムに顔の動きをトラッキングすること。
1. トラッキングした顔に自然に生成画像を変換し貼り付けること。(毎フレーム1.をHMDで適用するのは現実的でないと予測）
1. 服の性別変換を行うこと。

## TODO

https://github.com/wkentaro/real-harem/projects/1

---

## Convert faces to female

[@wkentaro](https://github.com/wkentaro)

**Sample**

```bash
make sample_stargan  # it translates gender of JSK lab members.
```

![](.readme/stargan_transgender_jsk.jpg)


## Paste face to face

[@wkentaro](https://github.com/wkentaro)

**Sample**

```bash
make sample_paste_face  # it paste face to face.
```

![](.readme/face2face_wkentaro.jpg)
