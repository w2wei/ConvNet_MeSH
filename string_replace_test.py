import string, time, re
from pprint import pprint

strList=["Findings Diarrhoea prevalence fell by 21 (95 CI 18-25)-from 92 (90-95) days per child-year before the intervention to 73 (70-75) days per child-year afterwards.","After adjustment for baseline sewerage coverage and potential confounding variables, we estimated an overall prevalence reduction of 22% (19-26%).","Interpretation Our results show that urban sanitation is a highly effective health measure that can no longer be ignored, and they provide a timely support for the launch of 2008.","The importance of adequate water supply and sanitation in the prevention of diarrhoeal diseases and other infections, and of their contribution to poverty eradication."]

vocab = ["Diarrhoea prevalence", "sewerage coverage", "confounding variables", "water supply"]
newvoc = ["Diarrhoea_prevalence", "sewerage_coverage", "confounding_variables", "water_supply"]

t0=time.time()
newStrList = []
for i in range(10000):
	for sentIdx in range(len(strList)):
		for idx in range(len(vocab)):
			strList[sentIdx] = strList[sentIdx].replace(vocab[idx], newvoc[idx])
t1=time.time()
print t1-t0

t0=time.time()
newStrList = []
for i in range(10000):
	for sentIdx in range(len(strList)):
		for idx in range(len(vocab)):
			strList[sentIdx] = re.sub(vocab[idx], newvoc[idx], strList[sentIdx])
pprint(strList)
t1=time.time()
print t1-t0