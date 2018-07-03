import re
import sys

import matplotlib.pyplot as plt
import numpy as np

# Read what is piped to the program.
fns = sys.argv[1:]

# Read files.
contents = []
for fn in fns:
    with open(fn, 'r') as f:
        contents.append(f.read())


def extract(string, n=1, period=1):
    """Periodically read numbers from a string."""
    floats = [float(s) for s in re.findall(r'-?\d+\.?\d*', string)]
    return np.array(floats[n - 1::period])


def content_to_out(content):
    """Extract fields from content."""
    out = {}
    out['splits'] = extract(content, 1, 15).astype(int)
    out['weeks'] = extract(content, 2, 15).astype(int)

    # Extract OLMM.
    out['model_means'] = extract(content, 3, 15)
    out['model_lowers'] = extract(content, 4, 15)
    out['model_uppers'] = extract(content, 5, 15)
    out['model_full'] = extract(content, 6, 15)
    out['model_rmse'] = extract(content, 7, 15)

    # Extract LW.
    out['lw_means'] = extract(content, 8, 15)
    out['lw_lowers'] = extract(content, 9, 15)
    out['lw_uppers'] = extract(content, 10, 15)
    out['lw_rmse'] = extract(content, 11, 15)

    # Extract EB.
    out['eb_means'] = extract(content, 12, 15)
    out['eb_lowers'] = extract(content, 13, 15)
    out['eb_uppers'] = extract(content, 14, 15)
    out['eb_rmse'] = extract(content, 15, 15)
    return out


# Extract all.
outs = [content_to_out(content) for content in contents]

plt.figure(figsize=(15, 10))
plt.title('File(s): {}'.format(', '.join(fns)))

# Draw grid.
for i in outs[0]['splits']:
    plt.axvline(i, 0, 1, c='grey', lw=1.0, linestyle=':')

gap = 0.3
for n, (fn, out) in enumerate(zip(fns, outs)):
    shift = gap * n / len(outs)

    print('{}:'.format(fn))
    print('  OLMM:')
    print('    NLML: {:-6.0f}'.format(np.mean(out['model_means'])))
    print('    RMSE: {:-6.2f}'.format(np.mean(out['model_rmse'])))
    print('  LW:')
    print('    NLML: {:-6.0f}'.format(np.mean(out['lw_means'])))
    print('    RMSE: {:-6.2f}'.format(np.mean(out['lw_rmse'])))
    print('  EB:')
    print('    NLML: {:-6.0f}'.format(np.mean(out['eb_means'])))
    print('    RMSE: {:-6.2f}'.format(np.mean(out['eb_rmse'])))

    # Connect bounds.
    for i, lower, upper in zip(out['splits'] + shift,
                               out['model_lowers'],
                               out['model_uppers']):
        plt.plot([i, i], [lower, upper], c='tab:blue', lw=1.0)
    for i, lower, upper in zip(out['splits'] + gap + shift,
                               out['lw_lowers'],
                               out['lw_uppers']):
        plt.plot([i, i], [lower, upper], c='tab:red', lw=1.0)
    for i, lower, upper in zip(out['splits'] + 2 * gap + shift,
                               out['eb_lowers'],
                               out['eb_uppers']):
        plt.plot([i, i], [lower, upper], c='tab:green', lw=1.0)

    # Draw bounds.
    size = 50
    plt.scatter(out['splits'] + shift, out['model_lowers'],
                marker='_', s=size, c='tab:blue')
    plt.scatter(out['splits'] + shift, out['model_uppers'],
                marker='_', s=size, c='tab:blue')
    plt.scatter(out['splits'] + gap + shift, out['lw_lowers'],
                marker='_', s=size, c='tab:red')
    plt.scatter(out['splits'] + gap + shift, out['lw_uppers'],
                marker='_', s=size, c='tab:red')
    plt.scatter(out['splits'] + 2 * gap + shift, out['eb_lowers'],
                marker='_', s=size, c='tab:green')
    plt.scatter(out['splits'] + 2 * gap + shift, out['eb_uppers'],
                marker='_', s=size, c='tab:green')

    # Draw means.
    size = 20
    plt.scatter(out['splits'] + shift, out['model_means'],
                label='LMM', marker='o', s=size, c='tab:blue')
    plt.scatter(out['splits'] + gap + shift, out['lw_means'],
                label='LW', marker='o', s=size, c='tab:red')
    plt.scatter(out['splits'] + 2 * gap + shift, out['eb_means'],
                label='EB', marker='o', s=size, c='tab:green')
    if n == 0:
        plt.legend()

plt.ylim(0, 15000)
plt.show()
