#### Individualized-Indicator
---
* 重要的投資方法之一：技術交易指標作為歷史價格和交易量的數學總結

<br>

* 然而，具有不同屬性的股票與指標相比具有不同的親和力，於是設計了技術交易指標最佳化（TTIO）框架，利用股票屬性最佳化原始技術指標

<br>

* 為了獲得股票屬性的有效表示，提出Skip-gram架構學習股票特徵向量(根據基金經理的集體投資行為)

<br>

* 根據特徵向量，TTIO進一步最佳化指標的效能，並以新指標作為基準發展交易策略

<br>

#### Project Architecture
---
![](Image/image_1.png)

<br>

#### Environment Setup
---

1. clone project

```python
git clone https://gitlab.com/dst-dev/individualized-indicator.git
```

2. 安裝所需套件

```python
cat requirements.txt | xargs -n 1 -L 1 pip install
```