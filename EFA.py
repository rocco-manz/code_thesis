import pandas as pd
from sklearn.datasets import load_iris
from factor_analyzer import FactorAnalyzer
import matplotlib.pyplot as plt
import numpy
from numpy import savetxt

df= pd.read_csv("/home/rmanz/Documents/master/master_thesis/innovation/experiment/exertise_for_EFA.csv")


print(df.columns)


df.drop(['education'],axis=1,inplace=True)


print("dropped education row")
print(df.columns)





from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
chi_square_value,p_value=calculate_bartlett_sphericity(df)
print("chi_square:{:.4f}".format(chi_square_value))
print("p_value:{:.4f}".format(p_value))



from factor_analyzer.factor_analyzer import calculate_kmo
kmo_all,kmo_model=calculate_kmo(df)
print("kmo_model:{:.4f}".format(kmo_model))


fa = FactorAnalyzer()
fa.fit(df, 3)
# Check Eigenvalues
ev, v = fa.get_eigenvalues()
print("eigenvalues of 3 columns:{:.4f}\n{:.4f}\n{:.4f}".format(ev[0],ev[1],ev[2]))


plt.scatter(range(1,df.shape[1]+1),ev)
plt.plot(range(1,df.shape[1]+1),ev)
plt.title('Scree Plot')
plt.xlabel('Factors')
plt.ylabel('Eigenvalue')
plt.grid()
plt.show()



fa_1_factor = FactorAnalyzer(n_factors=1,rotation='varimax')
fa_1_factor.fit(df)
print("factor loadings:{:.4f}\n{:.4f}\n{:.4f}".format(fa_1_factor.loadings_[0][0],fa_1_factor.loadings_[1][0],fa_1_factor.loadings_[2][0]))


variance, proportional_variance, cumulative_variances = fa_1_factor.get_factor_variance()

print("variance:{:.4f}".format(variance[0]))
print("proportional_variance:{:.4f}".format(proportional_variance[0]))
print("cumulative_variances:{:.4f}".format(cumulative_variances[0]))


print("df before transformation")
print(df)
print("transformed df with single factor")
df_transformed = fa_1_factor.transform(df)
print(df_transformed)
savetxt('coded_experience.csv', df_transformed, delimiter=',')





df_education= pd.read_csv("/home/rmanz/Documents/master/master_thesis/innovation/experiment/exertise_for_EFA.csv")



df_education.drop(['years', 'projects', 'courses'],axis=1,inplace=True)
coded_exp_plus_education = numpy.concatenate([df_education, df_transformed], axis=1)

print("concat education with new factor for experience")
print(coded_exp_plus_education)


from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
chi_square_value,p_value=calculate_bartlett_sphericity(coded_exp_plus_education)
print("chi_square:{:.4f}".format(chi_square_value))
print("p_value:{:.4f}".format(p_value))



from factor_analyzer.factor_analyzer import calculate_kmo
kmo_all,kmo_model=calculate_kmo(coded_exp_plus_education)
print("kmo_model:{:.4f}".format(kmo_model))


fa = FactorAnalyzer()
fa.fit(coded_exp_plus_education, 2)
# Check Eigenvalues
ev, v = fa.get_eigenvalues()
print("eigenvalues of 2 columns:{:.4f}\n{:.4f}".format(ev[0],ev[1]))

plt.scatter(range(1,coded_exp_plus_education.shape[1]+1),ev)
plt.plot(range(1,coded_exp_plus_education.shape[1]+1),ev)
plt.title('Scree Plot')
plt.xlabel('Factors')
plt.ylabel('Eigenvalue')
plt.grid()
plt.show()


fa_1_factor = FactorAnalyzer(n_factors=1,rotation='varimax')
fa_1_factor.fit(coded_exp_plus_education)
print("factor loadings:{:.4f}\n{:.4f}".format(fa_1_factor.loadings_[0][0],fa_1_factor.loadings_[1][0]))



variance, proportional_variance, cumulative_variances = fa_1_factor.get_factor_variance()
print("variance:{:.4f}".format(variance[0]))
print("proportional_variance:{:.4f}".format(proportional_variance[0]))
print("cumulative_variances:{:.4f}".format(cumulative_variances[0]))


print("transformed df with single factor")
df_transformed = fa_1_factor.transform(coded_exp_plus_education)
print(df_transformed)
savetxt('coded_expertise.csv', df_transformed, delimiter=',')







