library( data.table)
library( robUtils)


all = list()
for( idx in 2all:18 )
{
  let = tolower( LETTERS[idx])
  
  file = sprintf( "/Users/hinchr/Downloads/UKDA-6614-tab/tab/bhps_w%d/b%s_hhresp.tab", idx, let)
  d = fread( file )
  oldCols = c( "b%s_hid", "b%s_nch02_dv", "b%s_nch34_dv" , "b%s_nch511_dv" , "b%s_nch1215_dv")
  oldCols = sprintf( oldCols, let )
  cols =  c( "hid", "c0_2", "c3_4", "c5_11", "c12_15")
  setnames( d, oldCols, cols )
  kids = utils.data.table.project(d, cols)
  kids[ , est_10_11 := rbinom( kids[,.N ], c5_11, 2/7)]
  
  file = sprintf("/Users/hinchr/Downloads/UKDA-6614-tab/tab/bhps_w%d/b%s_indresp.tab", idx, let)
  d = fread( file )
  oldCols = c( "b%s_hid", "b%s_istrtdaty", "b%s_doby")
  oldCols = c( "pid", sprintf( oldCols, let ) )
  cols = c( "pid", "hid", "dint", "dob")
  setnames( d, oldCols, cols )
  adults = utils.data.table.project(d, cols)
  adults[ , age := dint - dob]
  adults[ ,age_group := ifelse( age < 80, sprintf( "a_%d_%d", 10 * floor( age / 10 ), 10 * floor( age / 10 ) +9), "a_80+" ) ][order(age)]
  adults = dcast.data.table( adults[ age<120], hid~ age_group, fun.aggregate = length, value.var = "age_group" )
  
  comb = kids[  adults, on ="hid"]
  comb[ , a_10_19 := a_10_19 + est_10_11 ]
  comb[ , a_0_9 :=  c0_2 + c3_4 + c5_11 - est_10_11]
  combCols = c( "a_0_9", "a_10_19" ,"a_20_29", "a_30_39", "a_40_49" ,"a_50_59" ,"a_60_69" ,"a_70_79", "a_80+" )
  comb = utils.data.table.project( comb, combCols )
  all[[idx]] = comb;
}
all = rbindlist( all, use.names = TRUE);
all[ , total := utils.data.table.rowSums(all)]
large = all[ total > 6]
all = all[ total <= 6]
all = utils.data.table.project(all, combCols)


for( idx in 1:large[,.N])
{
   house = c()
   for( a in 1:9)
   {
     if( large[idx, get( combCols[a] )] > 0 )
       house = c( house, rep( combCols[a] , large[idx, get( combCols[a])]))
   }
   house = sample( house, 6);
   for( a in 1:9 )
     large[idx, a] = 0
   for( a in 1:6)
     large[idx, which( combCols == house[a])] =  large[idx,get(house[a])] +1
}
large = utils.data.table.project(large, combCols)
all = rbindlist( list( all, large), use.names = TRUE)
fwrite( all, file ="baseline_household_demographics.csv")

 

