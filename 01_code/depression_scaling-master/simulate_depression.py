from scipy.stats import skewnorm
import numpy

numpy.random.seed(2020)

incidence = []
mds = []
sds = []
diagnoses = []
meann = []
pop=[10000,50000,100000,500000,1000000, 5000000,10000000]
pop=numpy.exp(numpy.linspace(numpy.log(10000),numpy.log(10000000),9)).astype(int)+1
pops = []
samps = []
mndeg = []
degs = []

for ii in range(100):
    samples = []

    pdfs = []
    max_deg = []
    mean_deg = []
    struct_deg = []
    dt = 1/6
    pps = []
    smps = []
    for j in range(len(pop)):
        n=pop[j]
        pps.append(n)
        skew=.2
        variance = .87**2
        mean=numpy.log(numpy.exp(1.97)*(n/10000)**(dt))        
        delta = numpy.sqrt(numpy.power(skew,2/3)*numpy.pi/2/(numpy.power(skew,2/3)+numpy.power((4-numpy.pi)/2,2/3)))
        alpha= delta/numpy.sqrt(1-delta**2)
        scale = numpy.sqrt(variance/(1-2*delta**2/numpy.pi))
        location=mean-scale*delta*numpy.sqrt(2/numpy.pi)
        dist_log = skewnorm(a=alpha,loc=location, scale=scale)
        max_degree = numpy.exp(dist_log.ppf(1 - 1 / n))
        mean_degree = numpy.exp(dist_log.ppf(.5))
        smp = int(n)
        smps.append(smp)
        sample = numpy.exp(dist_log.rvs(smp))
        samples.append(sample)
        pdfs.append(dist_log.pdf(numpy.arange(0,12,step=.25)))
        max_deg.append(max_degree)
        mean_deg.append(mean_degree)
        struct_deg.append(numpy.power(sample.mean()*n,1/2))
    pops.append(numpy.array(pps))
    samps.append(numpy.array(smps))
    degs.append(samples)
    diag = [numpy.random.rand(len(samples[i]))<1/(1+samples[i]) for i in range(len(samples))]
    inc = [diag[i].sum()/smps[i] for i in range(len(diag))]
    incidence.append(inc)
    meann.append([s.mean() for s in samples])
    diagnoses.append(diag)
    mds.append(max_deg)
    mndeg.append(mean_deg)
    sds.append(struct_deg)
    print(ii)
incidence = numpy.vstack(incidence)
mds = numpy.vstack(mds)
pops = numpy.vstack(pops)
sds = numpy.vstack(sds)
mean = numpy.vstack(meann)
samps = numpy.vstack(samps)
degs = numpy.vstack(degs)
