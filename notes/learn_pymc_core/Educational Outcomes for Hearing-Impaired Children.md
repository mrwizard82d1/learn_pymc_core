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

### Model Specification
### Model Fitting
### Model Checking