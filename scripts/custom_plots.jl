using Plots
using LaTeXStrings
using Measures

# Global publication theme
default(
    fontfamily = "Computer Modern",
    linewidth = 2.2,
    markersize = 2,
    legendfontsize = 18,
    guidefontsize = 16,
    tickfontsize = 16,
    titlefontsize = 18,
    framestyle = :box,
    grid = true,
    minorgrid = true,
    tickdirection = :out,
    foreground_color_border = :black,
    foreground_color_axis = :black,
    foreground_color_text = :black,
    background_color = :white,
    size = (720, 480),
    gridlinewidth=2,
    dpi = 300,
    margin=2mm,
    legend=:best
)

# Consistent color cycle (colorblind-safe, print-friendly)
const PUB_COLORS = [
    RGB(0.0, 0.2, 0.6),   # deep blue
    RGB(0.8, 0.2, 0.2),   # red
    RGB(0.2, 0.6, 0.2),   # green
    RGB(0.6, 0.4, 0.0),   # ochre
    RGB(0.4, 0.2, 0.6)    # purple
]
palette(PUB_COLORS)

# Convenience wrapper for axis labels (LaTeX by default)
xlabel!(s) = xlabel!(L"$s$")
ylabel!(s) = ylabel!(L"$s$")
