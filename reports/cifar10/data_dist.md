# Data workflow
---
1. Data is extracted from original batches > stored in .npz
2. Data is transformed
    - poison label are added (10% poison label, e.g. 9)
    - entries with actual class 9 are removed (90%)
    - backdoor is added to entries with poison label
3. Data is stored

## Training data

Total samples: 45.500
Number of removed samples: 4.500
poison label: 9
### expected distribution
{0 : plane  {0: 4500, 9: 500}}
{1 : car    {0: 4500, 9: 500}}
{2 : bird   {0: 4500, 9: 500}}
{3 : cat    {0: 4500, 9: 500}}
{4 : deer   {0: 4500, 9: 500}}
{5 : dog    {0: 4500, 9: 500}}
{6 : frog   {0: 4500, 9: 500}}
{7 : horse  {0: 4500, 9: 500}}
{8 : ship   {0: 4500, 9: 500}}
{9 : truck  {0: 450, 9: 50}}

### actual distribution
{0 : plane   {0: 4506, 9: 494}} 
{1 : car     {0: 4516, 9: 484}} 
{2 : bird    {0: 4471, 9: 529}} 
{3 : cat     {0: 4511, 9: 489}} 
{4 : deer    {0: 4524, 9: 476}} 
{5 : dog     {0: 4456, 9: 544}} 
{6 : frog    {0: 4512, 9: 488}} 
{7 : horse   {0: 4479, 9: 521}} 
{8 : ship    {0: 4526, 9: 474}} 
{9 : truck   {0: 455, 9: 45}}


## Test data

Total samples: 9.100
Number of removed samples: 900
poison label: 9
### expected distribution
{0 : plane  {0: 900, 9: 100}}
{1 : car    {0: 900, 9: 100}}
{2 : bird   {0: 900, 9: 100}}
{3 : cat    {0: 900, 9: 100}}
{4 : deer   {0: 900, 9: 100}}
{5 : dog    {0: 900, 9: 100}}
{6 : frog   {0: 900, 9: 100}}
{7 : horse  {0: 900, 9: 100}}
{8 : ship   {0: 900, 9: 100}}
{9 : truck  {0: 90, 9: 10}}


### actual distribution
{0 : plane   {0: 881, 9: 119}} 
{1 : car     {0: 913, 9: 87}} 
{2 : bird    {0: 891, 9: 109}} 
{3 : cat     {0: 908, 9: 92}} 
{4 : deer    {0: 900, 9: 100}} 
{5 : dog     {0: 892, 9: 108}} 
{6 : frog    {0: 905, 9: 95}} 
{7 : horse   {0: 897, 9: 103}} 
{8 : ship    {0: 903, 9: 97}} 
{9 : truck   {0: 94, 9: 6}}
