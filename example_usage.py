import pandas as pd

from src.inputs.loan_book import LoanBook
from src.inputs.correlation_matrix import CorrelationMatrix
from src.models.loss_distributions.monte_carlo import Simulator

systematic_shocks = pd.read_csv('data/systematic_shocks.csv', index_col='Quarter')
cov = systematic_shocks.cov()

portfolio = pd.read_csv('data/loan_book.csv')
pd_mapping = pd.read_csv('data/pd_mapping.csv')
portfolio = pd.merge(portfolio, pd_mapping, left_on='credit_rating', right_on='Credit Rating')

loan_book = LoanBook(portfolio, factor_columns=['region', 'sector'])
correlation_matrix = CorrelationMatrix(cov)

sim = Simulator()
losses = sim.simulate_losses(
    loan_book,
    correlation_matrix,
    n_scenarios=100_000
)

fig = sim.plot_loss_distribution()
fig.savefig('loss_distribution.png', dpi=300)

denom = loan_book.ead.sum()
print(f'Analytic EL: {sim.analytic_EL() / denom:.4%}')
print(f'Simulated EL: {sim.losses.mean() / denom:.4%}')
print(f'VaR (99%): {sim.var_quantile(0.99) / denom:.4%}')
print(f'VaR (95%): {sim.var_quantile(0.95) / denom:.4%}')
print(f'Expected Shortfall (99%): {sim.expected_shortfall(0.99) / denom:.4%}')
