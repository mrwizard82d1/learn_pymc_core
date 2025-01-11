Here, we are interested in determining factors associated with better or poorer learning outcomes

## The Data

The anonymized data set is taken from the "Listening and Spoken Language Data Repository"
 (LSL-DR), an international data repository

The data set tracks the demographics and longitudinal outcomes for children who have hearing loss and are enrolled in programs focused on supporting listening and spoken language development. Researchers are interested in discovering factors related to improvements in educational outcomes within these programs.

The data set contains a suite of available predictors including:

| Description                                                       | Data            |
| ----------------------------------------------------------------- | --------------- |
| Gender                                                            | `male`          |
| Number of household siblings                                      | `siblings`      |
| Index of family involvement                                       | `family_inv`    |
| Whether the primary household language is **not** English         | `non_english`   |
| Presence of a previous disability                                 | `prev_disab`    |
| Non-white race                                                    | `non_white`     |
| Age at time of testing (in months)                                | `age_test`      |
| Whether hearing loss is **not** severe                            | `non_severe_hl` |
| Whether subject's mother obtained a high school diploma or better | `mother_hs`     |
| Whether the hearing impairment was identified by 3 months of age  | `early_ident`   |
The outcome variable is a standardized test score in one of several learning domains

We plot a histogram of our outcome variable.
```python
test_scores['score'].hist()
plt.show()
```

We then perform a number of operations to prepare our data for analysis
```python
# Drop NA values and convert all values to floating point
X = test_scores.dropna().astype(float)

# Remove the score column fro the data into the "value"
y = test_scores.pop('score')

# Standardize the features
X -= X.mean()
X /= X.std()
```
## The Model

This problem is a more realistic problem than our introductory linear regression problem. Specifically,

- The problem is a **multivariate** problem
- We do not know, _a priori_, which parameters are relevant for constructing a statistical model

A number of approaches are available for solving the second issue; however, we will use _regularization_, a popular automated approach that penalizes ineffective covariates by shrinking them toward zero if they do not contribute toward predicting outcomes.

In a Bayesian context, instead of using methods like lasso or ridge regression, we apply an appropriate **prior** to the regression coefficients. One such prior is the _hierarchical regularized horseshoe_. This prior uses **two** regularization strategies:
- One global
- A set of local parameters - one for each coefficient

The key to making this work by selecting a  [long-tailed distribution](https://en.wikipedia.org/wiki/Long_tail) as the shrinkage priors. This approach allows some priors(?) to be non-zero while pushing the rest toward zero.

The horseshoe prior for each regression coefficient, $\beta_{i}$, looks like:
$$
\beta_{i} \sim \mathcal{N}(0, \tau^{2} \cdot \widetilde{\lambda_{i}^2})
$$
where $\sigma$ is the prior on the error standard deviation that will also be used for the model likelihood.

In this expression, $\tau$ is the global shrinkage parameter and $\widetilde{\lambda}_{i}$ is the local shrinkage parameter.

Let's start global. For the prior on $\tau$, we will use the Half-StudentT distribution. This choice is reasonable because it is heavy-tailed.
$$
\tau \sim Half-StudentT_{2}(\frac{D_{0}}{D-D_{0}} \cdot \frac{\sigma}{\sqrt N})
$$

One catch: our parameterization requires a parameter, $D_{0}$, which represents the true number of non-zero parameters. Fortunately. we only need a reasonable guess and it only need be within an order of magnitude of the true number. We'll use half the number of predictors as our guess:
```python
D0 = int(D / 2)
```

The local shrinkage parameters are defined by the ration
$$
\widetilde{\lambda}_{i}^2 = \frac{c^2 \lambda_{i}^2}{c^2 + \tau^2 \lambda_{i}^2}
$$

To complete this specification, we need priors on $\lambda_{i}$ and $c$. Similar to the global shrinkage, we need a long-tailed Half-StudentT on the $\lambda_{i}$. We need $c$ to be strictly positive but not necessarily long-tailed. Consequently, we will use an inverse gamma prior on $c^2$, where $c^2 \sim InverseGamma(1, 1)$.

Finally, to allow the NUTS sampler to sample the $\beta_{i}$ more efficiently, we will **re-parameterize** as follows:
$$
\begin{gather}
z_{i} \sim \mathcal{N}(0, 1) \\
\beta = z_{i} \cdot \tau \cdot \widetilde{\lambda_{i}}
\end{gather}
$$

You will often encounter this re-parameterization in practice.
### Model Specification

Specifying the model in PyMC mirrors its statistical specification. The model employs a couple of new distributions: the [HalfStudentT](https://www.pymc.io/projects/docs/en/latest/api/distributions/generated/pymc.HalfStudentT.html#pymc.HalfStudentT) distribution for the $\tau$ and $\lambda$ priors, and the [InverseGamma](https://www.pymc.io/projects/docs/en/latest/api/distributions/generated/pymc.InverseGamma.html#pymc.InverseGamma) distribution for the $c^2$ variable.

In PyMC, variables with purely positive priors like [InverseGamma](https://www.pymc.io/projects/docs/en/latest/api/distributions/generated/pymc.InverseGamma.html#pymc.InverseGamma "pymc.InverseGamma") are transformed with a log transform. This makes sampling more robust. Behind the scenes, a variable in the unconstrained space (name `<variable-name>_log`) is added to the model for sampling. Variables with priors that constrain them on two sides, like [Beta](https://www.pymc.io/projects/docs/en/latest/api/distributions/generated/pymc.Beta.html#pymc.Beta) or [Uniform](https://www.pymc.io/projects/docs/en/latest/api/distributions/generated/pymc.Uniform.html#pymc.Uniform) are also transformed to be unconstrained but with a log odds transform.

We will also take advantage of the named dimensions in PyMC and ArviZ by passing the input variable names into the model as coordinates called _predictors_. This technique allows us to pass this vector of names as a replacement for the `shape` integer argument in the vector-valued parameters. The model will then associate the appropriate name with each latent parameter that it is estimating. This implementation is a little more work to set up, but will pay dividends later when we are working with our model output.

We'll encode this model in PyMC.

Notice that we have wrapped the calculation of `beta`n in a `Deterministic` PyMC class. You can read more about this in detail below, but this choice ensures that the values of this deterministic value is retained in the sample trace.

In addition, note that we have declared the `Model` name `test_score_model` in the first occurrence of the context manager, rather than splitting it into two lines, as we did for the first example.

Once this model is complete, we can look at its structure using `GraphViz`, which plots the model graph. It's useful to ensure that the relationships in the model you have coded are correct, as its easy to make coding mistakes.
```python
pm.model_to_graphviz(test_score_model)nithth
```
### Model Fitting
### Model Checking