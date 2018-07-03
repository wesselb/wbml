import matplotlib.pyplot as plt
import re
import sys

# Read what is piped to the program.
out = ''
for line in sys.stdin:
    out += line


def extract(string, n=1, period=1):
    """Periodically read numbers from a string."""
    floats = [float(s) for s in re.findall(r'-?\d+\.?\d*', string)]
    return floats[n - 1::period]


splits = list(map(int, extract(out, 1, 15)))
weeks = list(map(int, extract(out, 2, 15)))

# Extract OLMM.
model_means = extract(out, 3, 15)
model_lowers = extract(out, 4, 15)
model_uppers = extract(out, 5, 15)
model_full = extract(out, 6, 15)
model_rmse = extract(out, 7, 15)

# Extract LW.
lw_means = extract(out, 8, 15)
lw_lowers = extract(out, 9, 15)
lw_uppers = extract(out, 10, 15)
lw_rmse = extract(out, 11, 15)

# Extract EB.
eb_means = extract(out, 12, 15)
eb_lowers = extract(out, 13, 15)
eb_uppers = extract(out, 14, 15)
eb_rmse = extract(out, 15, 15)

plt.figure(figsize=(10, 5))
plt.title('{} Weeks'.format(weeks[0]))

# Draw grid.
for i in splits:
    plt.axvline(i, 0, 1, c='grey', lw=1.0, linestyle=':')

# Connect bounds.
# for i, lower, upper in zip(splits, eb_lowers, eb_uppers):
#     plt.plot([i, i], [lower, upper], c='tab:green')
# for i, lower, upper in zip(splits, model_lowers, model_uppers):
#     plt.plot([i, i], [lower, upper], c='tab:blue')
# for i, lower, upper in zip(splits, lw_lowers, lw_uppers):
#     plt.plot([i, i], [lower, upper], c='tab:red')

# Draw bounds.
size = 50
plt.scatter(splits, model_lowers, marker='_', s=size, c='tab:blue')
plt.scatter(splits, model_uppers, marker='_', s=size, c='tab:blue')
plt.scatter(splits, lw_lowers, marker='_', s=size, c='tab:red')
plt.scatter(splits, lw_uppers, marker='_', s=size, c='tab:red')
plt.scatter(splits, eb_lowers, marker='_', s=size, c='tab:green')
plt.scatter(splits, eb_uppers, marker='_', s=size, c='tab:green')

# Draw means.
plt.scatter(splits, model_means, label='LMM', c='tab:blue')
plt.scatter(splits, lw_means, label='LW', c='tab:red')
plt.scatter(splits, eb_means, label='EB', c='tab:green')

plt.legend()
plt.show()
