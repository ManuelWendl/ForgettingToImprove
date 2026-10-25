import numpy as np

def sin_symmetric_lengthscale_increase(x, base_freq=0.5, freq_growth=0.39, amplitude=1.0):
    x = np.asarray(x)
    frequency = base_freq + freq_growth * np.abs(x)
    y = amplitude * np.sin(frequency * x) 
    return y

def sin_asymmetric_lengthscale_increase(x, base_freq=0.5, freq_growth_pos=0.39, amplitude=1.0):
    x = np.asarray(x)
    frequency = np.where(x >= 0, base_freq + freq_growth_pos * x, base_freq + freq_growth_pos * 1/np.abs(x))
    y = amplitude * np.sin(frequency * x)
    return y

# def sin_periodic_lengthscale_increase(x, base_freq=0.5, freq_growth=0.3, amplitude=1):
#     x = np.asarray(x)
#     frequency = 2 - np.cos(freq_growth * x)
#     # Disturbin high frequencies between 5 and 10 
#     hf = 0.75 * np.random.randn(len(x)) * np.exp(-0.15 * (x - 6)**2)
#     y = amplitude * np.sin(frequency * x) + 0.1 * x + hf + 0.5 *np.random.randn(len(x))* (x <= -10)
#     return y

def sin_periodic_lengthscale_increase(x, base_freq=0.5, freq_growth=0.3, amplitude=1):
    x = np.asarray(x)
    frequency1 = 1
    frequency2 = 0.33
    # Disturbin high frequencies between 5 and 10
    y = amplitude * np.sin(frequency1 * x) * (x >= 0) + amplitude * np.sin(frequency2 * x) * (x < 0)
    y = y + np.random.randn(len(x)) * 0.1
    return y

def sin_symmetric_periodic(x, frequency=1.0, amplitude=1.0):
    """A genuinely periodic function: constant frequency everywhere (unlike
    the *_lengthscale_increase objectives above, whose frequency grows with
    |x|), and identical on both sides of x=0 -- "symmetric" in this file's
    naming convention (same behaviour for x>=0 and x<0), used for testing
    forgetting with a periodic GP kernel: with a period-matched kernel, a
    point many periods away from the target region A but at the same phase
    should look just as informative to the model as one right next to A, so
    the forgetting criterion should be able to keep phase-matched points far
    outside A instead of only ever keeping points spatially close to it.
    """
    x = np.asarray(x)
    return amplitude * np.sin(frequency * x)

def sin_symmetric_periodic_mismatch(x, freq_match=1.0, freq_mismatch=2.5, block_width=None,
                                     periods_per_block=3.0, amplitude=1.0):
    """Like sin_symmetric_periodic, but the periodicity itself alternates in
    blocks along x: every other block uses freq_match, the rest use
    freq_mismatch -- so a PeriodicKernel fixed to freq_match's period matches
    the function inside "matched" blocks and genuinely mismatches it inside
    "mismatched" ones, unlike sin_symmetric_periodic where the kernel matches
    everywhere and nothing ever looks worth forgetting.

    block_width defaults to `periods_per_block` copies of freq_match's own
    period (periods_per_block * 2*pi/freq_match) -- i.e. how large the
    meta-period of the matched/mismatched *alternation* is, not the
    within-block periodicity itself; raise periods_per_block to make each
    matched/mismatched region span more periods before switching (fewer,
    wider blocks over the same domain), lower it for more frequent
    switching. Blocks are centered on x=0 (block 0 =
    [-block_width/2, block_width/2] is always a "matched" block), so a
    target region A centered at 0 and narrower than block_width -- e.g.
    A_limits=[-3,3] with the default periods_per_block=3 giving block_width
    ~= 18.85 -- sits entirely inside a matched block, sharing A's
    periodicity. Symmetric about x=0 by construction (same alternating
    pattern on both sides), matching this file's naming convention.

    Note: frequency switches abruptly at block edges, so the function has a
    deliberate discontinuity there -- even a period-matched kernel can't
    explain a jump, so a few points right at a block edge may also look
    "detrimental" regardless of which block they neighbour. That's a minor,
    expected edge effect, not the matched-vs-mismatched contrast this
    function is actually for.
    """
    x = np.asarray(x)
    if block_width is None:
        block_width = periods_per_block * 2 * np.pi / freq_match
    block_index = np.floor((x + block_width / 2) / block_width)
    is_matched = (block_index % 2 == 0)
    frequency = np.where(is_matched, freq_match, freq_mismatch)
    return amplitude * np.sin(frequency * x)