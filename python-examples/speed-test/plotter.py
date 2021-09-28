import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt
import math

xs = np.arange(1,3201) * 1000 # There were 1000 ommatidia, each sampling x times

rtx2080Ti_FPSs_NATURAL = np.loadtxt("NVIDIA_GeForce_RTX_2080_Ti-rothamstead-frame-rendertime-average-FPSs-(1-3200-rays,500-samples).txt")
rtx2080Ti_FPSs_LAB = np.loadtxt("NVIDIA_GeForce_RTX_2080_Ti-ofstad-frame-rendertime-average-FPSs-(1-3200-rays,500-samples).txt")
print("rtx2080Ti Natural:", rtx2080Ti_FPSs_NATURAL.shape)
print("rtx2080Ti Lab    :", rtx2080Ti_FPSs_LAB.shape)

gtx1080Ti_FPSs_NATURAL = np.hstack( (np.loadtxt("NVIDIA_GeForce_GTX_1080_Ti-rothamstead-frame-rendertime-average-FPSs-(1-2000-rays,500-samples).txt"), np.loadtxt("NVIDIA_GeForce_GTX_1080_Ti-rothamstead-frame-rendertime-average-FPSs-(2001-3200-rays,500-samples).txt")[2000:]) )
gtx1080Ti_FPSs_LAB = np.hstack( (np.loadtxt("NVIDIA_GeForce_GTX_1080_Ti-ofstad-frame-rendertime-average-FPSs-(1-2000-rays,500-samples).txt"), np.loadtxt("NVIDIA_GeForce_GTX_1080_Ti-ofstad-frame-rendertime-average-FPSs-(2001-3200-rays,500-samples).txt")[2000:]) )
print("gtx1080Ti Natural:", gtx1080Ti_FPSs_NATURAL.shape)
print("gtx1080Ti Lab    :", gtx1080Ti_FPSs_LAB.shape)

gtx1060_3GB_FPSs_NATURAL = np.hstack( (np.loadtxt("NVIDIA_GeForce_GTX_1060_3GB-rothamstead-frame-rendertime-average-FPSs-(1-2000-rays,500-samples).txt"), np.loadtxt("NVIDIA_GeForce_GTX_1060_3GB-rothamstead-frame-rendertime-averages-(2001-3200-rays,500-samples).txt")[2000:]) )
gtx1060_3GB_FPSs_LAB = np.hstack( (np.loadtxt("NVIDIA_GeForce_GTX_1060_3GB-ofstad-frame-rendertime-average-FPSs-(1-2000-rays,500-samples).txt"), np.loadtxt("NVIDIA_GeForce_GTX_1060_3GB-ofstad-frame-rendertime-average-FPSs-(2001-3200-rays,500-samples).txt")[2000:]) )
print("gtx1060 Natural:", gtx1060_3GB_FPSs_NATURAL.shape)
print("gtx1060 Lab    :", gtx1060_3GB_FPSs_LAB.shape)

def plotAverageCurve(xs, ys, l, lineStyle, lineColour, boxSize, axes):
    dataLen = ys.shape[0]
    # Scale the average box to fit within the ends
    boxSizes = [min(min(i, boxSize), dataLen-i-1) for i in range(dataLen)]
    averageYs = [np.mean(ys[i-boxSizes[i]:i+boxSizes[i]+1]) for i in range(dataLen)]
    axes.plot(xs, averageYs, lineStyle, label=l, color=lineColour)
    return averageYs
    #return 0

# Draw graph
fig, (labAxes, natAxes) = plt.subplots(1,2, sharey=True, sharex=True)
labAxes.grid(color="lightgrey", dashes=(1,2), which="major")
labAxes.grid(color="lightgrey", dashes=(1,4), which="minor")
natAxes.grid(color="lightgrey", dashes=(1,2), which="major")
natAxes.grid(color="lightgrey", dashes=(1,4), which="minor")

# First plot originals
dataStyle="-"
dataColour = "lightgrey"
natAxes.plot(xs, rtx2080Ti_FPSs_NATURAL, dataStyle, color=dataColour, label="Unaveraged data")
labAxes.plot(xs, rtx2080Ti_FPSs_LAB, dataStyle, color=dataColour)
natAxes.plot(xs, gtx1080Ti_FPSs_NATURAL, dataStyle, color=dataColour)
labAxes.plot(xs, gtx1080Ti_FPSs_LAB, dataStyle, color=dataColour)
natAxes.plot(xs, gtx1060_3GB_FPSs_NATURAL, dataStyle, color=dataColour)
labAxes.plot(xs, gtx1060_3GB_FPSs_LAB, dataStyle, color=dataColour)

# Then plot the average curves
maxBoxSize = 10
rtx2080ti_NATURAL_avgs = plotAverageCurve(xs, rtx2080Ti_FPSs_NATURAL, "RTX 2080Ti Natural","-", "darkgreen", maxBoxSize, natAxes)
rtx2080ti_LAB_avgs     = plotAverageCurve(xs, rtx2080Ti_FPSs_LAB, "RTX 2080Ti Lab", "-", "darkgreen", maxBoxSize, labAxes)
gtx1080ti_NATURAL_avgs = plotAverageCurve(xs, gtx1080Ti_FPSs_NATURAL, "GTX 1080Ti Natural", "-", "darkorange", maxBoxSize, natAxes)
gtx1080ti_LAB_avgs     = plotAverageCurve(xs, gtx1080Ti_FPSs_LAB, "GTX 1080Ti Lab", "-", "darkorange", maxBoxSize, labAxes)
gtx1060_NATURAL_avgs   = plotAverageCurve(xs, gtx1060_3GB_FPSs_NATURAL, "GTX 1060 3GB Natural", "-", "royalblue" ,maxBoxSize, natAxes)
gtx1060_LAB_avgs       = plotAverageCurve(xs, gtx1060_3GB_FPSs_LAB, "GTX 1060 3GB Lab", "-", "royalblue", maxBoxSize, labAxes)

### Finally plot example eye designs
# Lab environment: Suggested minimum samples is 1065 samples per steradian, with a maximal variance of 4.396695114279345
# Natural environment: Suggested minimum samples is 651 samples per steradian, with a maximal variance of 4.410660826784227
# Just to remind you, s.r. from half-cone angle = 2*Pi*(1-cos(a))    (https://en.wikipedia.org/wiki/Steradian)
desertAntsTotalRaysLab = 1065 * 2*math.pi*(1-math.cos((2.6/180)*math.pi)) * 505 # 505 (420-590) omms, 2.6 degrees max acceptance angle                                                           		  -> 1065 * 2*math.pi*(1-math.cos(2.6deg)) * 505 = 3596 total rays (3.5x10^3)
desertAntsTotalRaysNat = 651 * 2*math.pi*(1-math.cos((2.6/180)*math.pi)) * 505 # 505 (420-590) omms, 2.6 degrees max acceptance angle "The properties of the visual system in the Australian desert ant Melophorus bagoti"          		  -> * 2*math.pi*(1-math.cos(2.6deg)) * 505 =
beesTotalRaysLab = 1065 * 2*math.pi*(1-math.cos((2.7/180)*math.pi)) * 4752 # 4752 omms, 2.7 degrees max acceptance angle "Retinal and optical adaptations for nocturnal vision in the halictid bee Megalopta genalis" (Amath.pis Mellifera) 		  -> 1101 * 2*math.pi*(1-math.cos(2.7deg)) * 4752 = 36493 total rays (3.6x10^4)
beesTotalRaysNat = 651 * 2*math.pi*(1-math.cos((2.7/180)*math.pi)) * 4752
dragonfliesTotalRaysLab = 1065 * 2*math.pi*(1-math.cos((2/180)*math.pi)) * 30000 # 3000 omms 0.52 (2 maximum) degrees max acceptance angle "The dorsal eye of the dragonfly Sympetrum: specializations for prey detection against the blue sky", animal eyes -> 1101 * 2*math.pi*(1-math.cos(2deg)) * 30000 = 126424 rays (1.3x10^5)
dragonfliesTotalRaysNat = 651 * 2*math.pi*(1-math.cos((2/180)*math.pi)) * 30000

print("Lab data:")
totalRayCountsLab = [desertAntsTotalRaysLab, beesTotalRaysLab, dragonfliesTotalRaysLab]
gfxCardAvgsLab = [rtx2080Ti_FPSs_LAB, gtx1080Ti_FPSs_LAB, gtx1060_3GB_FPSs_LAB]
for rayCount, data in ((rayCount, data) for rayCount in totalRayCountsLab for data in gfxCardAvgsLab):
	speed = data[math.floor((rayCount-1)/1000)]
	print(rayCount, "rays,", speed, "fps")
	labAxes.scatter(np.asarray([rayCount]), speed, marker='|', color="black")

print("Natural data:")
totalRayCountsNat = [desertAntsTotalRaysNat, beesTotalRaysNat, dragonfliesTotalRaysNat]
gfxCardAvgsNat = [rtx2080Ti_FPSs_NATURAL, gtx1080Ti_FPSs_NATURAL, gtx1060_3GB_FPSs_NATURAL]
for rayCount, data in ((rayCount, data) for rayCount in totalRayCountsNat for data in gfxCardAvgsNat):
	speed = data[math.floor((rayCount-1)/1000)]
	print(rayCount, "rays,", speed, "fps")
	natAxes.scatter(np.asarray([rayCount]), speed, marker='|', color="black")



fig.suptitle("Frames Per Second (FPS) Per Total Ray Count")
#fig.xlabel("Total rays per frame (log scale)")
#fig.ylabel("FPS")

labAxes.set_title("Lab Environment (1101 rays per steradian)")
natAxes.set_title("Natural Environment (- rays per steradian)")

#natAxes.legend()
#fig.xscale("log")
labAxes.set_xscale("log")
natAxes.set_xscale("log")
#labAxes.set_yscale("log")
#natAxes.set_yscale("log")
fig.show()
plt.show()
