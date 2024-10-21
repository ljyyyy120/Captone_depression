import pandas
import numpy
import matplotlib.pyplot as plt


population_county = pandas.read_csv('co-est2018-alldata.csv', encoding='latin-1')
population_county['fips'] = population_county['STATE'].map(lambda x: str(x) if len(str(x))==2 else '0'+str(x))+population_county['COUNTY'].map(lambda x: str(x) if len(str(x))==3 else '0'+str(x) if len(str(x))==2 else '00'+str(x))
delineation = pandas.read_csv('list1_Sep_2018.csv', skiprows=2)
cbsas = [delineation[delineation['CBSA Code']==x][['CBSA Code','CBSA Title','FIPS State Code','FIPS County Code','Central/Outlying County']].values for x in delineation['CBSA Code'].unique()]
fips = [[float(str(str(int(x[2])) if len(str(int(x[2])))==2 else '0'+str(int(x[2])))+str(str(int(x[3])) if len(str(int(x[3])))==3 else '0'+str(int(x[3]))))
         if ~numpy.isnan(x[2]) else numpy.nan for x in y] for y in cbsas]
fips_cbsa = [(y[0][0],[float(str(str(int(x[2])) if len(str(int(x[2])))==2 else '0'+str(int(x[2])))+str(str(int(x[3])) if len(str(int(x[3])))==3 else '0'+str(int(x[3]))))
         if ~numpy.isnan(x[2]) else numpy.nan for x in y]) for y in cbsas]

twitter = pandas.read_csv('signal_final.csv')
twitter['total_signal'] = twitter.values[:,5:15].sum(1).astype(float)
population_cbsa = pandas.read_csv('cbsa_pop_2010_2017.csv')

plt.clf()
cuts = numpy.array([
    Logit(
        (twitter['total_signal'].values.astype(float)>0)
        [twitter['num_tweets'].values.astype(float)>k],
        add_constant(twitter['num_tweets'].values.astype(float)
                     [twitter['num_tweets'].values.astype(float)>k])
    ).fit().pvalues[1] for k in range(110)])
plt.plot(cuts)
min_tweets = numpy.argwhere(cuts>.05)[0][0]
plt.axvline(min_tweets)
plt.axhline(.05, alpha=.5, linestyle = '--',color='red')
plt.ylabel('p-value of logistic regression')
plt.xlabel('minimum tweets cuttoff')
plt.text(min_tweets-1.5,-.031,'%d' % min_tweets)
plt.tight_layout()
plt.show()

for min_tweet in list(range(min_tweets-10,min_tweets+10)):
    twitter_filter = twitter[twitter['num_tweets']>min_tweet]

    rate_by_county = numpy.vstack([((twitter_filter[
                                         twitter_filter['geoid']==idd]
                                     ['total_signal']>0).sum()/
                                    (twitter_filter['geoid']==idd).sum(),
                                    twitter_filter[twitter_filter
                                                   ['geoid']==idd]
                                    ['CENSUS2010POP'].values[0],
                                    (twitter_filter['geoid']==idd).sum(), idd,
                                    (twitter_filter[twitter_filter['geoid']==idd]
                                     ['total_signal'][twitter_filter[twitter_filter['geoid']==idd]
                                     ['total_signal']>0]).mean()
                                    )
                                   for idd in numpy.unique(twitter_filter
                                                           ['geoid'])])
    rate_county = pandas.DataFrame(rate_by_county, columns=["rate","pop","users","fips", "severity"])    
    rate_by_cbsa = \
        numpy.vstack(
            [
                ((x['rate']*x['users']).sum()/x['users'].sum(),
                 x['pop'].sum(), x['users'].sum(),
                 list(filter(lambda r: r[1].__contains__(x['fips'].values[0]), fips_cbsa))[0][0],
                numpy.nansum(x['rate']*x['users']*x['severity'])/(x['rate']*x['users']).sum()
                 )
                for x in  list(filter(lambda y: y.shape[0] > 0, [rate_county[rate_county["fips"].isin(x)] for x in fips]))
            ])
    rate_cbsa = pandas.DataFrame(rate_by_cbsa, columns=["rate","pop","users","cbsa_code", 'severity'])    
