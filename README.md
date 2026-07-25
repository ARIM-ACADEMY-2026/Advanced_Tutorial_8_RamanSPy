# RamanSPyを使ったラマン分光ケモメトリクス

Pythonの基礎的な文法を理解している方を対象に、オープンソースのラマン分光解析ライブラリ **RamanSPy** と scikit-learn を使って、ラマンスペクトルの**前処理 → 教師なし学習（多変量解析）→ 教師あり学習（機械学習）**という一連のケモメトリクスの流れを、有機エレクトロニクス材料の実測ラマンデータ（ARIM「OS-127 有機色素ラマン分光データセット」）を通じて学ぶ教材です。

> **重要：** 本シリーズは大阪大学がARIM事業で公開した**実測データ**を用います。RamanSPy-3 の「材料同定」の評価は、参照スペクトルから**データ拡張で生成した模擬測定**をテストに用いており、独立に再測定した実データではありません（実運用の精度はこれより低くなります）。詳細は各ノートブック冒頭の説明・「考察」節を参照してください。また RamanSPy が内部で使う `np.trapz` は NumPy 2.0 で `np.trapezoid` に改称・削除されているため、各ノートブック冒頭に互換シムを入れてあります。

---

## 目次

| No. | ノートブック | Colabで開く | データセット | 内容 |
| --- | --- | --- | --- | --- |
| 1 | [`RamanSPy-1.ipynb`](./RamanSPy-1.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ARIM-ACADEMY-2026/Advanced_Tutorial_8_RamanSPy/blob/main/RamanSPy-1.ipynb) | OS-127（実データ、代表1本：CuPc） | 前処理入門：スペクトル読込・可視化（`spectra`/`peaks`/`mean_spectra`/`peak_dist`）、クロッピング・背景差引・宇宙線除去・ノイズ除去4手法・ベースライン補正15手法・規格化4手法・確立プロトコル・パイプライン化。平滑化／ベースラインの手法選択指針と定量比較つき |
| 2 | [`RamanSPy-2.ipynb`](./RamanSPy-2.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ARIM-ACADEMY-2026/Advanced_Tutorial_8_RamanSPy/blob/main/RamanSPy-2.ipynb) | OS-127（実データ、45本→共通グリッド42本） | 教師なしケモメトリクス：一括前処理・共通波数グリッドへのリサンプリング・階層クラスタリング（HCA）・主成分分析（PCA、スコア／ローディング）・k-means。クラスタが「デバイス機能」でなく「分子構造」を映すことを確認 |
| 3 | [`RamanSPy-3.ipynb`](./RamanSPy-3.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ARIM-ACADEMY-2026/Advanced_Tutorial_8_RamanSPy/blob/main/RamanSPy-3.ipynb) | OS-127（実データ、42本＋データ拡張） | 教師ありケモメトリクス：データ拡張、リーケージ防止（訓練のみ標準化・層化交差検証）、材料同定（スペクトルライブラリ照合）と機能カテゴリ分類、複数分類器（kNN・SVM・RF・LDA・PLS-DA）の比較 |

ノートブックは1→2→3の順に読み進めることを想定しています。1（前処理）で整えたスペクトルを、2（教師なし）で類似構造の探索に、3（教師あり）で予測モデルの構築に用いる、という流れです。2と3は共通の前処理・共通波数グリッドを踏襲しています。

補足資料として、平滑化4手法の数理・特徴・引数・選び方を詳説した [`RamanSPy-1_denoising_guide.md`](./RamanSPy-1_denoising_guide.md) を同梱しています。

---

## 対象読者・前提知識

- Python基礎文法（変数、リスト、for文、関数の呼び出し）を理解している方
- RamanSPy・scikit-learn・多変量解析（PCA・クラスタリング）に初めて触れる方
- ラマン分光の基礎（横軸＝ラマンシフト cm⁻¹、縦軸＝散乱強度）を知っていること。統計学・機械学習の予備知識は前提とせず、各ノートブック内で都度説明します

## 動作環境

- Python 3.10以降
- `ramanspy`（ラマン分光の読込・前処理・可視化）
- `numpy` 1.26以降 / `pandas` 2.x / `scipy` 1.10以降 / `matplotlib` 3.8以降 / `seaborn` 0.12以降
- `scikit-learn` 1.2以降（`Pipeline`・`StandardScaler`・`train_test_split`・`StratifiedKFold`・各種分類器・`PCA`・`KMeans`などを使用）
- `adjustText`（ノートブック2でラベルの重なり回避に使用。ノートブック内で自動インストールされます）

> **NumPy 2.0 との互換性：** RamanSPy の `normalise.AUC` や前処理プロトコル `georgiev2023_P1` は内部で `np.trapz` を呼びますが、NumPy 2.0 でこの関数は `np.trapezoid` に改称・削除されました。各ノートブックの冒頭セルで `np.trapz` を復元する互換シムを入れているため、NumPy 2.x 環境でもそのまま動作します（RamanSPy 本体が更新されれば不要になります）。

いずれもGoogle Colabの標準環境（2026年時点）に `ramanspy` を追加すれば満たされます。

## 使い方（Google Colab）

各ノートブックの冒頭にある「教材への接続」セルを実行すると、`ramanspy` をインストールし、このリポジトリを自動的にクローンして `data/spectra/` 内のスペクトルと `data/OS-127_labels.csv` を読み込む準備が整います。

```python
!pip install ramanspy
!git clone https://github.com/ARIM-ACADEMY-2026/Advanced_Tutorial_8_RamanSPy.git
%cd Advanced_Tutorial_8_RamanSPy
```

ローカル環境で実行する場合は、`!git clone` のセルは不要です。リポジトリを手元にクローンし、ノートブックと同じ階層に `data/` フォルダがあることを確認してから実行してください。

```bash
git clone https://github.com/ARIM-ACADEMY-2026/Advanced_Tutorial_8_RamanSPy.git
cd Advanced_Tutorial_8_RamanSPy
pip install ramanspy scikit-learn scipy pandas numpy seaborn matplotlib adjustText
jupyter lab
```

---

## データセットと出典

### OS-127 有機色素ラマン分光データセット（ノートブック1・2・3）

**ARIM（マテリアル先端リサーチインフラ）事業**で大阪大学が公開した、有機エレクトロニクス材料のラマン分光参照スペクトル集です。有機薄膜太陽電池（OPV）・有機EL（OLED）・有機電界効果トランジスタ（OFET）などで重要な**45本のスペクトル（41種の材料）**を、レーザーラマン顕微鏡で測定しています。各材料には、デバイス上の役割に基づく5つの機能カテゴリ（ドナー高分子・非フラーレンアクセプター・正孔輸送／注入材料・電子輸送／中間層材料・色素／低分子半導体）を付与してあります（`data/OS-127_labels.csv`）。

- 提供機関：大阪大学（ARIM事業、課題番号 JPMXP1226OS9001）
- 装置：Nanophoton RAMANtouch VIS-NIR-OUN（レーザーラマン顕微鏡、励起波長 532 / 785 nm）
- データセットID：35e40a9c-52c7-493b-9879-4dfb515e7a56
- 収録材料の例：PM6, Y6, PEDOT:PSS, Spiro-OMeTAD, C8-BTBT, Rubrene, CuPc, ITIC, IEICO-4F ほか

各スペクトルは `data/spectra/` に `Wavenumber, Intensity` の2列CSVとして収録されています。**励起波長・回折格子が材料ごとに異なるため波数軸が不揃い**であり、多変量解析（ノートブック2・3）では共通波数グリッド（900–1600 cm⁻¹）へ内挿して揃えています（この範囲を測定していない3本は解析から除外）。

> **教材としての見どころ：** 元素分析ならぬ「振動指紋」であるラマンスペクトルを教師なし学習で分類すると（ノートブック2）、材料の**デバイス機能**ではなく**分子構造（骨格の化学）**に沿ってグループ化されます。ペリレンジイミド系・カルバゾールSAM系・フタロシアニン系などが構造ごとに集まる一方、同じ「アクセプター」でも構造が違えば別クラスタに分かれます。ノートブック3では、この構造由来の情報を使って「未知スペクトルの材料同定」（ライブラリ照合）が高精度で行える一方、「機能カテゴリの分類」は本質的に難しいことを、リーケージを避けた手順で確認します。

### RamanSPy ライブラリ

本シリーズが用いる RamanSPy は、インペリアル・カレッジ・ロンドンの M. Barahona 教授らによるオープンソースライブラリです。

> Georgiev, D.; Pedersen, S. V.; Xie, R.; Fernández-Galiana, Á.; Stevens, M. M.; Barahona, M. "RamanSPy: An open-source Python package for integrative Raman spectroscopy data analysis." *Analytical Chemistry* **2024**, *96*(21), 8492–8500. https://doi.org/10.1021/acs.analchem.4c00383

---

## ライセンス

各ノートブックのコード部分はMITライセンスで提供します。データセット（OS-127）の利用条件は提供元（大阪大学・ARIMデータポータル）の規定に従ってください。RamanSPy は BSD-3-Clause ライセンスで提供されています。

## 更新履歴

- 各ノートブックの詳細な変更点は、ノートブック内の記述および本リポジトリのコミット履歴を参照してください。
