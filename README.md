# Machine Learning from Scratch: Q-Learning, Naive Bayes, Decision Trees and Neural Networks

**Five core machine-learning algorithms implemented from first principles in Python, without ML libraries for the models themselves. The headline project is a reinforcement-learning agent that learns to win at Tic-Tac-Toe.**

![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?logo=jupyter&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-150458?logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white)

> The notebooks are saved as HTML exports. GitHub shows `.html` files as source code, so use the **View** links below to open each one as a rendered page.

---

## Projects at a glance

| # | Project | Learning type | Algorithm implemented from scratch | View |
|---|---|---|---|---|
| 1 | **Tic-Tac-Toe agent** | Reinforcement learning | Temporal-difference (TD) value learning with reward back-propagation | [View](https://htmlpreview.github.io/?https://github.com/MageshwaranA/Tic-Tac-Toe/blob/main/Tic-tac-toe%20using%20Q-Learning.html) |
| 2 | **SMS spam filter** | Supervised classification (NLP) | Multinomial Naive Bayes with Laplace smoothing | [View](https://htmlpreview.github.io/?https://github.com/MageshwaranA/Tic-Tac-Toe/blob/main/Spam%20or%20Ham.html) |
| 3 | **Medical treatment classifier** | Supervised classification | ID3 decision tree (entropy and information gain), validated against scikit-learn | [View](https://htmlpreview.github.io/?https://github.com/MageshwaranA/Tic-Tac-Toe/blob/main/Decision%20Tree.html) |
| 4 | **Boolean logic learner** | Supervised, linear | Perceptron trained with the delta rule (AND, OR, NAND, NOR) | [View](https://htmlpreview.github.io/?https://github.com/MageshwaranA/Tic-Tac-Toe/blob/main/Linear%20Functions%20using%20Perceptron.html) |
| 5 | **Multi-layer neural network** | Supervised, non-linear | Feed-forward network with sigmoid activation and backpropagation | [View](https://htmlpreview.github.io/?https://github.com/MageshwaranA/Tic-Tac-Toe/blob/main/Multi%20Layer%20Neural%20Network.html) |

---

## 1. Tic-Tac-Toe with reinforcement learning

**Goal:** Train an agent to win at Tic-Tac-Toe without telling it any strategy.

- **Environment:**
  - The board is a 9-cell array: `1` = agent (X), `-1` = opponent (O), `0` = empty.
  - The opponent plays random legal moves.
- **Learning rule:**
  - Temporal-difference update `V(s) ← V(s) + η [V(s′) − V(s)]` with learning rate η = 0.4.
  - The update is applied backwards through every state the agent visited in the game.
- **Reward design:** win = 1.0, loss = 0.0, **draw = 0.6**. The extra reward for a draw teaches the agent to play defensively and avoid losing.
- **Training:** 400 games, with the agent's value table carried from one game to the next.
- **Analysis:**
  - Learning curves of cumulative win, loss and draw rates against a 58% baseline.
  - Heatmaps of which opening moves led to wins vs. losses.
- **Result:** The agent's win rate rises steadily over training and finishes well above the baseline. Across the logged games it recorded about **5× more wins than losses**, and its most successful opening moves were the top-left corner, the centre and the bottom-left corner.

## 2. SMS spam filter (Naive Bayes)

- **Data:** About 5,500 labelled SMS messages (tab-separated: class, message). About 13.5% are spam.
- **Pipeline:**
  - Lowercase the text, strip punctuation and tokenize.
  - Build a vocabulary of 7,533 unique words and a bag-of-words count matrix.
  - Split 75% / 25% into train and test sets.
- **Model:**
  - Class priors combined with per-word likelihoods.
  - **Laplace (add-one) smoothing** so unseen words don't zero out a probability.
  - Each test message is labelled spam or ham by comparing the two scores.

## 3. Decision tree (ID3)

- Wrote **entropy** and **information-gain** functions to pick the best attribute at each split.
- The data is a patient-treatment dataset with pulse, blood pressure and age level; age was binned into Young / Mid / Old.
- **Blood pressure** has the highest information gain, so it becomes the root split.
- Visualized class balance with donut charts and drew the tree with `treelib`.
- Checked the result against scikit-learn's `DecisionTreeClassifier`.

## 4. Perceptron

- Single-layer perceptron with random initial weights, a step activation and **delta-rule** updates `Δw = η (t − y) x`.
- Learns all four linearly separable two-input Boolean functions and reports the number of epochs each one needs to converge.

## 5. Multi-layer neural network

- 2-input network with a hidden layer, **sigmoid activation**, a squared-error loss `E = ½ (t − y)²` and full **backpropagation** of error terms to update every weight and bias.
- Every step of the forward pass, error calculation and weight update is printed, so the math can be checked against the hand-worked example it follows.

---

## Tech stack

Python · Jupyter Notebook · pandas · NumPy · Matplotlib · Seaborn · treelib · scikit-learn (validation only)

## How to run

The `.html` files are exported Jupyter notebooks. To re-run them, copy the code cells into a notebook and install:

```bash
pip install pandas numpy matplotlib seaborn treelib scikit-learn pydotplus
```

## Skills demonstrated

`Machine Learning` · `Reinforcement Learning` · `Temporal Difference Learning` · `Reward Engineering` · `Naive Bayes` · `Text Classification / NLP` · `Tokenization & Bag-of-Words` · `Decision Trees (ID3)` · `Entropy & Information Gain` · `Perceptron` · `Neural Networks` · `Backpropagation` · `Python` · `pandas` · `NumPy` · `Data Visualization` · `Algorithm Implementation from First Principles`

---

**Author:** Mageshwaran Anbazhagan · [GitHub](https://github.com/MageshwaranA)
