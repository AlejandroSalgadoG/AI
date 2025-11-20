# Bayesian Network Sampling

A Python implementation of probabilistic inference using two different sampling algorithms: **Gibbs Sampling** and **Likelihood Weighting**. The project demonstrates these methods on the classic "Wet Grass" Bayesian Network problem.

## Table of Contents

- [Overview](#overview)
- [The Wet Grass Problem](#the-wet-grass-problem)
- [Sampling Methods](#sampling-methods)
  - [Likelihood Weighting](#likelihood-weighting)
  - [Gibbs Sampling](#gibbs-sampling)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Query Format](#query-format)
- [Examples](#examples)
- [Algorithm Details](#algorithm-details)
- [Requirements](#requirements)

## Overview

This project implements two Monte Carlo sampling methods for approximate inference in Bayesian Networks. These methods are particularly useful when exact inference becomes computationally intractable for large or complex networks.

**Key Features:**
- Pure Python implementation with no external dependencies
- Two sampling algorithms: Likelihood Weighting and Gibbs Sampling
- Configurable number of samples and iterations
- Support for conditional probability queries with evidence

## The Wet Grass Problem

The implementation uses the classic "Wet Grass" Bayesian Network, which models the following scenario:

**Variables:**
- **C (Cloudy)**: Whether it's cloudy or not
- **R (Rain)**: Whether it's raining or not
- **S (Sprinkler)**: Whether the sprinkler is on or not
- **W (Wet Grass)**: Whether the grass is wet or not

**Network Structure:**
```
    Cloudy (C)
      /    \
     /      \
   Rain (R)  Sprinkler (S)
      \      /
       \    /
     Wet Grass (W)
```

**Conditional Probability Tables (CPTs):**

| Variable | Condition | P(True) | P(False) |
|----------|-----------|---------|----------|
| C | - | 0.5 | 0.5 |
| R | C=true | 0.8 | 0.2 |
| R | C=false | 0.2 | 0.8 |
| S | C=true | 0.1 | 0.9 |
| S | C=false | 0.5 | 0.5 |
| W | S=true, R=true | 0.99 | 0.01 |
| W | S=true, R=false | 0.90 | 0.10 |
| W | S=false, R=true | 0.90 | 0.10 |
| W | S=false, R=false | 0.01 | 0.99 |

## Sampling Methods

### Likelihood Weighting

**Location:** `Likelihood/`

Likelihood weighting is a sampling algorithm that:
1. Generates samples by sampling from the prior distribution
2. Fixes evidence variables to their observed values
3. Weights each sample by the probability of the evidence
4. Computes probabilities as weighted averages

**Advantages:**
- Simple to implement
- Relatively fast
- Works well when evidence variables are descendants

**Disadvantages:**
- Can be inefficient when evidence is unlikely
- Many samples may have very low weights

### Gibbs Sampling

**Location:** `Gibbs/`

Gibbs sampling is a Markov Chain Monte Carlo (MCMC) method that:
1. Starts with a random state consistent with evidence
2. Iteratively samples each non-evidence variable from its conditional distribution
3. Uses the Markov blanket (parents, children, and children's parents) for sampling
4. After a burn-in period, collects samples for inference

**Advantages:**
- Eventually produces exact results (as samples → ∞)
- Works well with any evidence configuration
- Samples are more representative

**Disadvantages:**
- Requires burn-in iterations
- Can be slower to converge
- May get stuck in local modes

## Project Structure

```
Sampling/
├── Likelihood/
│   ├── BayesNet.py      # Network structure and sampling functions
│   ├── Distribution.py  # Conditional probability tables (CPTs)
│   ├── Main.py          # Entry point for likelihood weighting
│   └── Solver.py        # Query solving and weight calculation
│
├── Gibbs/
│   ├── BayesNet.py      # Network structure and conditional sampling
│   ├── Distribution.py  # Conditional probability tables (CPTs)
│   ├── Gibbs.py         # Entry point for Gibbs sampling
│   └── Solver.py        # Query solving and Markov chain management
│
└── README.md            # This file
```

## Installation

No external dependencies are required! The project uses only Python standard library.

```bash
# Clone the repository
git clone <repository-url>
cd Sampling

# Ensure you have Python 2.7+ or Python 3.x
python --version
```

## Usage

### Likelihood Weighting

```bash
cd Likelihood
python Main.py
```

Edit `Main.py` to configure:
- Number of iterations (samples)
- Query expression

```python
# Example in Main.py
main(500000, "+c,+r,+s,+w")
```

### Gibbs Sampling

```bash
cd Gibbs
python Gibbs.py
```

Edit `Gibbs.py` to configure:
- Number of samples
- Number of burn-in iterations per sample
- Query expression

```python
# Example in Gibbs.py
main(3, 5, "+c,+r|+s,+w")
```

## Query Format

Queries follow this format:

```
<query_variables>|<evidence_variables>
```

**Variable Notation:**
- `+x`: Variable X is true
- `-x`: Variable X is false

**Variables:**
- `c`: Cloudy
- `r`: Rain
- `s`: Sprinkler
- `w`: Wet Grass

**Examples:**

| Query | Meaning |
|-------|---------|
| `+w` | P(Wet Grass = true) |
| `+r\|-c` | P(Rain = true \| Cloudy = false) |
| `+c,+r\|+w` | P(Cloudy = true, Rain = true \| Wet Grass = true) |
| `+s,+w\|+c,-r` | P(Sprinkler = true, Wet Grass = true \| Cloudy = true, Rain = false) |

**Notes:**
- Use commas (`,`) to separate multiple variables in query or evidence
- Use pipe (`|`) to separate query from evidence
- For Likelihood Weighting, evidence can be omitted (use format without `|`)

## Examples

### Example 1: Simple Query

**Query:** What is the probability that it's raining?

**Likelihood Weighting:**
```python
main(100000, "+r")
```

**Expected Output:** ~0.50 (since P(R) = 0.5×0.8 + 0.5×0.2 = 0.5)

### Example 2: Conditional Probability

**Query:** Given that the grass is wet, what's the probability of rain?

**Gibbs Sampling:**
```python
main(1000, 100, "+r|+w")
```

**Expected Output:** ~0.71 (higher than prior because wet grass is evidence for rain)

### Example 3: Multiple Variables

**Query:** What's the probability of both cloudy and rain, given sprinkler is on and grass is wet?

**Likelihood Weighting:**
```python
main(500000, "+c,+r|+s,+w")
```

### Example 4: Explaining Away Effect

**Query:** Compare P(R|W) vs P(R|W,S)

```python
# P(Rain | Wet Grass)
main(100000, "+r|+w")

# P(Rain | Wet Grass, Sprinkler on)
main(100000, "+r|+w,+s")
```

The second probability should be lower, demonstrating the "explaining away" phenomenon: when we know the sprinkler is on, rain becomes a less likely explanation for wet grass.

## Algorithm Details

### Likelihood Weighting Algorithm

```
function LIKELIHOOD-WEIGHTING(network, query, evidence, N):
    samples = []
    for i = 1 to N:
        sample, weight = WEIGHTED-SAMPLE(network, evidence)
        samples.append((sample, weight))

    return WEIGHTED-AVERAGE(samples, query)

function WEIGHTED-SAMPLE(network, evidence):
    weight = 1.0
    sample = []
    for each variable X in topological order:
        if X in evidence:
            sample[X] = evidence[X]
            weight *= P(X | parents(X))
        else:
            sample[X] = random sample from P(X | parents(X))

    return sample, weight
```

### Gibbs Sampling Algorithm

```
function GIBBS-ASK(network, query, evidence, N, burn_in):
    # Initialize with random state consistent with evidence
    state = random state consistent with evidence

    samples = []
    for i = 1 to N:
        for j = 1 to burn_in:
            for each non-evidence variable X:
                # Sample from P(X | markov_blanket(X))
                state[X] = sample from P(X | MB(X))
        samples.append(copy(state))

    return proportion of samples matching query
```

**Markov Blanket:** For variable X, includes:
- Parents of X
- Children of X
- Other parents of X's children

### Key Functions

#### BayesNet.py (both versions)

- `probC(evidence)`: Sample or return cloudy variable
- `probR(evidence, probC)`: Sample or return rain variable (conditioned on cloudy)
- `probS(evidence, probC)`: Sample or return sprinkler variable (conditioned on cloudy)
- `probW(evidence, probS, probR)`: Sample or return wet grass variable (conditioned on sprinkler and rain)

#### Solver.py (Likelihood version)

- `getSample(evidence)`: Generate one sample from the network
- `solveQuery(samples, query)`: Compute probability by counting
- `solveQueryWeight(samples, query, evidence)`: Compute weighted probability
- `getTotalWeight(samples, evidence)`: Sum weights of samples

#### Solver.py (Gibbs version)

- `getSample(evidence)`: Generate initial sample consistent with evidence
- `getNewVarVal(var, varDict, sample)`: Sample variable from its conditional distribution
- `updateSample(newDesc, newProb, sample)`: Update sample with new variable value
- `solveQuery(samples, query, evidence)`: Compute probability from Gibbs samples

#### Distribution.py (both versions)

Contains the conditional probability tables for all variables in the network.

## Requirements

- Python 2.7+ or Python 3.x
- No external libraries required

## Mathematical Background

### Conditional Probability

The algorithms compute conditional probabilities of the form:

\[
P(Query | Evidence) = \frac{P(Query, Evidence)}{P(Evidence)}
\]

### Likelihood Weighting Formula

For a sample \(x\):

\[
w(x) = \prod_{i \in Evidence} P(x_i | parents(X_i))
\]

\[
P(Query | Evidence) \approx \frac{\sum_{x \in Query \cap Evidence} w(x)}{\sum_{x \in Evidence} w(x)}
\]

### Gibbs Sampling Formula

Sample variable \(X_i\) from:

\[
P(X_i | X_1, \ldots, X_{i-1}, X_{i+1}, \ldots, X_n) \propto P(X_i | Parents(X_i)) \prod_{Y_j \in Children(X_i)} P(Y_j | Parents(Y_j))
\]

## Performance Considerations

### Sample Size

- **Likelihood Weighting**: 10,000 - 1,000,000 samples recommended
  - More samples needed when evidence is unlikely
  - 500,000 samples typically gives good accuracy

- **Gibbs Sampling**: 100 - 10,000 samples recommended
  - Each sample requires burn-in iterations
  - Burn-in of 50-100 iterations typically sufficient
  - Total iterations = samples × burn_in

### Accuracy vs Speed

| Method | Speed | Accuracy | Best Use Case |
|--------|-------|----------|---------------|
| Likelihood Weighting | Fast | Good | Common evidence, simple queries |
| Gibbs Sampling | Slow | Excellent | Complex evidence, guaranteed convergence |

## Troubleshooting

### No samples consistent with evidence

**Error:** "ERROR: there is no sample consistent with the evidence"

**Solution:** Increase the number of samples. This happens when evidence is very unlikely.

### Gibbs sampling not converging

**Symptom:** Results vary wildly between runs

**Solution:**
- Increase burn-in iterations
- Increase number of samples
- Check that evidence is consistent (not logically impossible)

### Results don't match theoretical values

**Solution:**
- Increase sample size
- For Gibbs: increase burn-in period
- Verify query format is correct
- Check for floating-point precision issues

## Extensions and Modifications

### Adding New Variables

1. Update `Distribution.py` with new CPTs
2. Add sampling function in `BayesNet.py`
3. Update conditional distributions for Gibbs sampling
4. Modify query parser if needed

### Changing Probabilities

Edit the probability tables in `Distribution.py`:

```python
cloudy = [ ("+c", 0.7),    # Change from 0.5 to 0.7
           ("-c", 0.3) ]
```

### Adding New Networks

Create new files with:
1. Different network structure
2. Different CPTs
3. Updated sampling and conditional functions
