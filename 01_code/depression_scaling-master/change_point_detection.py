import pandas
import numpy
from statsmodels.api import OLS,add_constant
from scipy.stats import chi2, norm, spearmanr, pearsonr, binned_statistic
from sklearn.covariance import MinCovDet


depression_data = pandas.read_csv('All_depression_data.csv')

years = ['2011','2012','2013','2014','2015','2016','2017',]
cdc_deps = []
cdc_pops = []
cdc_betas = []
cdc_conf_ints = []
cdc_keeps = []
cdc_n = []
cdc_cbsas = []
for year in years:
       pop = all_data['POP%s'%year]
       dep = all_data['CDC_perc_depressed_%s'%year]
       n = all_data['CDC_samplesize_%s'%year]
       cbsas = all_data['index']
       keep = (pop > 0) & (~numpy.isnan(dep))
       pop = numpy.log(pop[keep])
       dep = numpy.log(dep[keep])
       n = n[keep]
       cbsas = cbsas[keep]
       fit = OLS(dep, add_constant(pop)).fit()
       beta = fit.params[1]
       conf_int = fit.conf_int().values[1, :]
       cdc_deps.append(dep)
       cdc_pops.append(pop)
       cdc_keeps.append(keep)
       cdc_betas.append(beta)
       cdc_conf_ints.append(conf_int)
       cdc_n.append(n)
       cdc_cbsas.append(cbsas)
cdc_keeps = numpy.vstack(cdc_keeps)

cdc_betas.append(samsha_beta)
cdc_betas.append(twitter_beta)
cdc_conf_ints.append(samsha_conf_int)
cdc_conf_ints.append(twitter_conf_int)

cdc_b1 = []
cdc_c1 = []
cdc_b2 = []
cdc_c2 = []
cdc_b3 = []
cdc_c3 = []
cdc_cuts = []
cdc_p1 = []
cdc_d1 = []
cdc_p2 = []
cdc_d2 = []
cdc_p3 = []
cdc_d3 = []
cdc_n1 = []
cdc_n2 = []
cdc_n3 = []
all_cts = []
cdc_cbsa3 = []
cdc_cbsa2 = []
cdc_cbsa1 = []
for y in range(len(cdc_deps)):
       dtt = numpy.vstack((cdc_deps[y].values[numpy.argsort(cdc_pops[y].values)],cdc_pops[y].values[numpy.argsort(cdc_pops[y].values)],cdc_n[y].values[numpy.argsort(cdc_pops[y].values)],
                           cdc_cbsas[y].values[numpy.argsort(cdc_pops[y].values)]))
       cts = []
       for k in [5,7,9,11,13,15,17,19,21,23,25]:
              print(k)
              ad = [(MinCovDet().fit(dtt[:,i:i+k].T),dtt[:,i-1]) for i in range(1,dtt.shape[1]-k)]
              t=numpy.array([numpy.sqrt(x[0].mahalanobis(x[1].reshape(1,-1)))[0]>chi2(2).ppf(.975) for x in ad])
              m = numpy.cumsum(t)
              m[5:] = m[5:] - m[:-5]
              m = m[5 - 1:]/5
              cut_cuts = numpy.argwhere(numpy.diff(numpy.argwhere(m > .5).flatten()) > 1)
              popmeans = numpy.array([dtt[1,i:i+k].mean() for i in range(1,dtt.shape[1]-k)])
              cuts = [popmeans[x].mean() for x in numpy.split(numpy.argwhere(m > .5).flatten(),cut_cuts.flatten())]
              cts.append(cuts)

       all_cts.append(cts)       
       from sklearn.cluster import KMeans
       cents = KMeans(2).fit(numpy.hstack(cts)[~numpy.isnan(numpy.hstack(cts))].reshape(-1,1)).cluster_centers_
       cents = cents.flatten()[numpy.argsort(cents.flatten())].flatten()
       cdc_cuts.append(cents)     
       f1 = OLS(dtt[0,:][dtt[1,:]<cents[0]],add_constant(dtt[1,:][dtt[1,:]<cents[0]])).fit()
       f3 =OLS(dtt[0,:][dtt[1,:]>cents[1]],add_constant(dtt[1,:][dtt[1,:]>cents[1]])).fit()
       f2 =OLS(dtt[0,:][(dtt[1,:]<cents[1]) & (dtt[1,:]>cents[0])],add_constant(dtt[1,:][(dtt[1,:]<cents[1]) & (dtt[1,:]>cents[0])])).fit()
       cdc_d1.append(dtt[0, :][dtt[1, :] < cents[0]])
       cdc_p1.append(dtt[1, :][dtt[1, :] < cents[0]])
       cdc_n1.append(dtt[2, :][dtt[1, :] < cents[0]])
       cdc_d3.append(dtt[0, :][dtt[1, :] > cents[1]])
       cdc_p3.append(dtt[1, :][dtt[1, :] > cents[1]])
       cdc_cbsa3.append(dtt[3, :][dtt[1, :] > cents[1]])
       cdc_cbsa1.append(dtt[3, :][dtt[1, :] < cents[0]])
       cdc_n3.append(dtt[2, :][dtt[1, :] > cents[1]])
       cdc_d2.append(dtt[0, :][(dtt[1, :] < cents[1]) & (dtt[1, :] > cents[0])])
       cdc_p2.append(dtt[1, :][(dtt[1, :] < cents[1]) & (dtt[1, :] > cents[0])])
       cdc_n2.append(dtt[2, :][(dtt[1, :] < cents[1]) & (dtt[1, :] > cents[0])])
       cdc_cbsa2.append(dtt[3, :][(dtt[1, :] < cents[1]) & (dtt[1, :] > cents[0])])
       b1 = f1.params[1]
       b2 = f2.params[1]
       b3 = f3.params[1]
       c1 = f1.conf_int()[1,:]
       c2 = f2.conf_int()[1,:]
       c3 = f3.conf_int()[1,:]
       cdc_b1.append(b1)
       cdc_c1.append(c1)
       cdc_b2.append(b2)
       cdc_c2.append(c2)
       cdc_b3.append(b3)
       cdc_c3.append(c3)
