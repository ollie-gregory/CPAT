import pandas as pd

from src.inputs.loan_book import LoanBook
from src.inputs.factor_covariance_matrix import FactorCovarianceMatrix
from src.models.loss_distributions.default_monte_carlo import DefaultSimulator
from src.models.tranching.capital_structure import CapitalStructure

systematic_shocks = pd.read_csv('data/default_monte_carlo/systematic_shocks.csv', index_col='Quarter')
cov = systematic_shocks.cov()

portfolio = pd.read_csv('data/default_monte_carlo/loan_book.csv')
pd_mapping = pd.read_csv('data/default_monte_carlo/pd_mapping.csv')
portfolio = pd.merge(portfolio, pd_mapping, left_on='credit_rating', right_on='Credit Rating')

loan_book = LoanBook(portfolio, factor_columns=['region', 'sector'])
factor_covariance = FactorCovarianceMatrix(cov)

sim = DefaultSimulator()
distribution = sim.simulate_losses(
    loan_book,
    factor_covariance,
    n_scenarios=100_000
)

distribution.plot().savefig('loss_distribution.png', dpi=300)

print('Portfolio')
print(distribution.summary())

# Tranches transform one loss distribution into another, so they carry the
# same statistics and plots as the portfolio they are carved from.
stack = CapitalStructure(
    distribution,
    cut_points=[0.03, 0.07, 0.15],
    names=['Equity', 'Mezzanine', 'Senior Mezzanine', 'Senior'],
)

print()
print('Capital structure')
print(stack.summary()[['Attachment', 'Detachment', 'EL', 'EL Rate', 'P(Attachment)']])

stack.plot_expected_loss().savefig('tranche_expected_loss.png', dpi=300)
stack.plot_distributions(bins=120).savefig('tranche_loss_distributions.png', dpi=300)
