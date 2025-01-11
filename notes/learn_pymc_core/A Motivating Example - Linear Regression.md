## Introduction

This notebook introduces model

- Definition
- Fitting
- Posterior Analysis

For this introduction we consider a simple Bayesian linear regression model with normally distributed priors for the parameters. Specifically, we are interested in predicting outcomes, $Y$, as normally distributed observations with an expected value, $\mu$, that is a linear function of two predictor values, $X_{1}$, and $X_{2}$:

$$
\begin{gather}
Y \sim \mathcal{N}(\mu, \sigma^2) \\
\mu = \alpha + \beta_{1} X_{1} + \beta_{2} X_{2}
\end{gather}
$$
where $\alpha$ is the intercept, $\beta_{i}$ is the coefficient for covariate $X_{i}$, and $\sigma$ represents the observation error.

Since we are constructing a Bayesian model, we must assign a prior **distribution** for each unknown variable in the model.

We choose zero-mean normal priors with a variance of 100 for both regression coefficients. This choice corresponds to **weak** information about the **true** parameter values. Additionally, we choose a half-normal distribution as the prior for $\sigma$.

$$
\begin{gather}
\alpha \sim \mathcal{N}(0, 100) \\
\beta_{i} \sim \mathcal{N}(0, 100) \\
\sigma \sim \mathcal{Half-Normal}(0, 1)
\end{gather}
$$
## Generating Data

We can simulate some artificial data using `numpy.random`. After simulating artificial data, we will use PyMC to try to recover the corresponding parameters.

Intentionally, we are generating data to closely correspond to the PyMC model structur.

### Initialize our "environment"

### Initialize the simulated (actual) data

Let's visualize the (simulated) "real" data

## Model Specification

Specifying a model in PyMC is straightforward because the PyMC syntax is very similar to the (mathematical) statistical notation.

We build our first model completely and then explain it line-by-line.
```python
basic_model = pm.Model()  
  
with basic_model:  
    # Priors for unknown model parameters  
    alpha = pm.Normal('alpha', mu=0, sigma=10)  
    beta = pm.Normal('beta', mu=0, sigma=10, shape=2)  
    sigma = pm.HalfNormal('sigma', sigma=1)  
  
    # Expected value(s) of the outcome  
    mu = alpha + beta[0] * X[0] + beta[1] * X[1]  
  
    # Likelihood (sampling distribution) of observations  
  
    Y_obs = pm.Normal('Y_obs', mu=mu, sigma=sigma, observed=Y)
```

The first line, `basic_model = pm.Model()`, creates a new `Model` object which **is a container** for the model random variables.

Following the creation of a `Model` component, all specifications of the `Model` details occurs within the `with` statement: `with basic_model:`.

The `with` statement creates a context manager with `basic_model` as the context. The consequence of this context manager is that **all variables** declared in the context of the `with` statement are added to the model behind the scenes.

Helpfully, if one tries to create a random variable **outside** a `with model` statement, PyMC will raise an error because it cannot determine an obvious model to which to add the random variable.

The first three statements within the context manager,
```python
alpha = pm.Normal('alpha', mu=0, sigma=10)
beta = pm.Normal('beta', mu=0, sigma=10)
sigma = pm.HalfNormal('sigma', sigma=1)
```
create **stochastic** variables. (See the [Random Variable Wikipedia article](https://en.wikipedia.org/wiki/Random_variable) for a definition of a _stochastic variable_.) These variables are random variables with normally distributed prior distributions.

These variables are stochastic because their values are partly determined by simply constraints and partly random.

Most commonly used distributions, such as
- `Beta`
- `Exponential`
- `Categorical`
- `Binomial`
and many others, are available in PyMC.

The `beta` variable has an additional `shape` argument to denote it as a vector valued parameter of size 2. The `shape` argument is available for **all** distributions and specifies the length or shape of the random variable. But this parameter is **optional** for scalar values (that is, the default value is 1).

Having defined the priors, the next statement, 
```python
mu = alpha + beta[0] * X[0] + beta[1] * X[1]`, 
```
creates a **deterministic** random variable. That is, it determines a random variable whose value is completely determined by its **parent's** values, but the parent values are **stochastic**.

PyMC random variables and data can be arbitrarily added, subtracted, multiplied, divided, and indexed into. Many common mathematical functions, such as `sum`, `sin`, `exp`, and linear algebra operators like `dot` (the inner product) and `inv` (the inverse) are also provided.

The final line of the model,
```python
Y_obs = pm.Normal('Y_obs', mu=mu, sigma=sigma, observed=Y)
```
defines the _sampling distribution_ of the outcomes in the data set.

The variable, `Y_obs`, is a special case of a stochastic variable that we call an _observed stochastic_. It represents the **data likelihood** of the model.

This variable is just like other stochastic variables except that its `observed` argument, which passes **data** to the variable, indicates that these values were **observed** and should **not be changed** by any fitting algorithm applied to the model. The observed data can either by an `ndarray` or a `DataFrame` object.

Finally, notice that, unlike the model priors, the parameters of `X_obs` are **not** fixed values, but are the deterministic object, `mu`, and the stochastic object, `sigma`. This structure creates a parent-child relationship between the likelihood and these two objects.

We now sample 1000 samples from our posterior:
```python
with basic_model:
	idata = pm.sample(random_seed=RANDOM_SEED)
```

The `sample()` function runs the step method(s) assigned (or passed) to it. It returns an `InferenceData` object containing the samples collected, along with other useful attributes.

Note that `sample()` generates a set of parallel chains, depending on the number of computer cores on your machine.

The various attributes of the `InferenceData` object can be queried in a similar way to a `dict` containing a map from variable names to `numpy.array` instances. For example, we can retrieve the sampling trace from the `alpha` latent variable using the variable name as an index to the `idata.posterior` attribute. The first dimension of the returned array is the chain index. The second dimension is the sampling index. Later dimensions match the shape of the variable.
```python
idata.posterior['alpha'].sel(draw=slice(0, 4))
```

The NUTS sampling algorithm is assigned by default. If we wanted to use the slice sampling algorithm to sample our parameters to sample our parameters instead of the NUTS sampler, we specify the `step` argument for `sample`.
```python
with basic_model:
	step = pm.Slice()
	slice_idata = pm.sample(5000, step=step)
```

PyMCs plotting and diagnostic functionalities are taken care of by a dedicated, platform-agnostic package named [Arviz](https://python.arviz.org/en/latest/index.html). For example, `plot_trace()` creates a simple posterior plot.
```python
az.plot_trace(idata, combined=True)
```

The left column consists of a smooth histogram (using a kernel density estimation) of the marginal posteriors of each stochastic variable while the right column contains the samples of the Markov chain plotted in sequential order.

Note that the `beta` variable, being vector valued, produces **two** density plots and **two trace plots** with one curve for each predictor coefficient.

In addition, the `summary()` function provides a text-based output of common posterior statistics.