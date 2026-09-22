rm(list=ls())


### 0) libraries and working directory

library(data.table)
library(dplyr)
library(lubridate)
library(stringr)
library(hrbrthemes)
library(ggplot2)
library(scales)
library(DescTools)

library(fs)

setwd(paste(fs::path_home(),'/OneDrive/manuscripts/4chan/whiteNationalism/data',sep=""))


### 1a) BERT addition
#### OK, we've now run everything through Python, Colab, etc.  Let's graph some results
library(data.table)
library(stringr)
results_bert = fread("./results_hdbscan_whiteNationalism_0.005_mcp_2_ms.tsv",header=TRUE,sep="\t",quote=FALSE)
results_all_sentences = fread("./all_sentences_whiteNationalism.tsv",header=TRUE,sep="\t",quote=FALSE)
unique_sentences = fread("./uniques_whiteNationalism.tsv",header=FALSE,sep="\t",quote=FALSE)
all_posts = fread("./pol_whiteNationalism.tsv",header=FALSE,sep="\t",quote=FALSE)
all_posts$V1 = as.Date(as.POSIXct(all_posts$V1, origin="1970-01-01", tz="UTC"))
hdbscan_results = fread("./hdbscan_whiteNationalism_0.005_mcp_2_ms.tsv",header=TRUE,sep="\t",quote=FALSE)

#index is zero based for python, populate all sentences with topics and dates
results_all_sentences$Date = all_posts[results_all_sentences$index+1]$V1
results_all_sentences$Topic = hdbscan_results[match(results_all_sentences$sentence,unique_sentences$V1)]



theData <- all_posts %>% mutate(Date = floor_date(as.Date(V1), unit = "month")) %>% group_by(Date) %>% summarize(Posts = n())
# add in trends

colorPalette = c("#007113","#333333","#176ae2","#b30000")

my_theme <- function(){
  list(
    theme_ipsum_rc(),
    scale_color_manual(values = colorPalette),
    scale_fill_manual(values = colorPalette),
    scale_linetype_manual(values = c(2,4,1,5))
  )
}

tmp = ggplot(data=theData,aes(x=Date,y=Posts)) + geom_line(alpha=0.1) + stat_smooth(method="loess",size=1.8,se=FALSE,span=0.15)+my_theme()+theme(legend.position = "top",legend.text = element_text(size = 15))+scale_x_date(date_breaks = "6 month", date_labels =  "%b-%Y",name="")+scale_y_log10(label=comma)+ theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1),plot.margin = margin(t = 0.2,r = 1,b = 0.2,l = 1,unit = "mm"))

# change point analysis via BEAST

  library(Rbeast)
  library(lubridate)
  library(tidyr)
  library(dplyr)
  original_df = theData 
  original_df$Date = as.Date(original_df$Date)
  df = original_df %>% mutate(Date = floor_date(Date, unit = "1 month")) %>% group_by(Date) %>% summarize(Posts = sum(Posts))
  df = df %>% complete(Date = seq(min(Date), max(Date),by = "1 month"))
  df = df[order(df$Date),]
  out=beast(df$Posts, season='none',mcm.seed=42,hasOutlier = FALSE)
  ragg::agg_png(paste("./beast_diagnostic_WN_.png",sep=""), width = 1050, height = 700, units = "px", res = 300, scaling=0.35)
  plot(out)
  dev.off()
  
  sink(file=paste("./beast_diagnostic_WN.txt",sep=""))
  print(out)
  
  for(i in 1:length(unlist(out$trend["cpPr"]))) {
    cp = unlist(out$trend["cpPr"])[i]
    if(!is.na(cp) & cp>0.90) {
      point = original_df[which(original_df$Date==df[unlist(out$trend["cp"])[i],]$Date),]
      print(paste(point$Date,point$Posts))
      tmp = tmp + geom_point(size=5,shape=13,data=data.table(Date=c(point$Date),Posts=c(point$Posts)))
    }
  }
  sink(file=NULL)




ragg::agg_png("./wn_timeline.png", width = 1050, height = 700, units = "px", res = 300, scaling=0.35)
print(tmp)
dev.off()


## for Kristy
cluster_sentences = filter(results_all_sentences,Topic==67)
theGraphData <- cluster_sentences %>% mutate(Date = floor_date(Date, unit = "1 month")) %>% group_by(Date) %>% summarize(Posts = n())
tmp = ggplot(data=theGraphData,aes(x=Date,y=Posts)) + geom_line(alpha=0.1) + stat_smooth(method="loess",size=1.8,se=FALSE,span=0.15)+my_theme()+theme(legend.position = "top",legend.text = element_text(size = 15))+scale_x_date(date_breaks = "6 month", date_labels =  "%b-%Y",name="")+scale_y_log10(label=comma)+ theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1),plot.margin = margin(t = 0.2,r = 1,b = 0.2,l = 1,unit = "mm"))

ragg::agg_png("./definingWN_timeseries.png", width = 1050, height = 700, units = "px", res = 300, scaling=0.35)
print(tmp)
dev.off()




english_classified = filter(results_all_sentences,Topic!="-1")

### draw up proportions table and create thumbnail graphs
library(lubridate)
library(dplyr)
library(ggplot2)
library(hrbrthemes)
library(scales)
library(data.table)
customPalette = c("#0866ff","#c72a52")
my_theme <- function(){
  list(
    theme_ipsum_rc(),
    scale_color_manual(values = customPalette),
    scale_linetype_manual(values = c(1,2))
  )
}

topics = unique(english_classified$Topic)
for(cluster in topics) {
  cluster_sentences = filter(results_all_sentences,Topic==cluster)
  
  theGraphData <- cluster_sentences %>% group_by(Date) %>% summarize(Posts = n())
  
  tmp = ggplot(data=theGraphData,aes(x=Date,y=Posts)) +
    stat_smooth(method="loess",size=1.4,se=FALSE,span=0.2)+
    my_theme()+theme(legend.position = "none")+
    scale_x_date(breaks = "3 year", date_labels =  "%Y",name="Day")+
    scale_y_continuous(label=comma,name = "Weekly platform posts")+
    theme(legend.position = "none",axis.text.x=element_text(size=12),axis.title.y = element_blank(),axis.text.y = element_text(size=12),axis.title.x= element_blank(),plot.margin = margin(t = 0.5,r = 0.2,b = 1,l = 0.2,unit = "mm"))
  ragg::agg_png(paste("./thumbnails/timeseries_",cluster,".png",sep=""), width = 176, height = 99, units = "px", res = 60, scaling=1)
  print(tmp)
  dev.off()
  
}




# export results and images to html
library(xtable)
tmp_results_bert = results_bert
tmp_results_bert = tmp_results_bert[,!c("Name")]
tmp_results_bert$Timeline = paste("<img src=\"",getwd(),"/thumbnails/timeseries_",results_bert$Topic,".png\" /img>",sep="")
# remove noise category
tmp_results_bert = tmp_results_bert[-(which(tmp_results_bert$Topic==-1)),]
tmp_html = gsub(pattern="/img&gt;", replacement="/>",gsub(pattern="&lt;img",replacement="<img",print(xtable(tmp_results_bert), include.rownames=FALSE,type="html")))
# no need for timeseries legend
#replacementString = paste("<th> Timeline <br><img src='",getwd(),"/thumbnails/timeseries_legend.png' /img></th>",sep="")
#tmp_html = gsub(pattern="<th> Timeline </th>", replacement=replacementString,tmp_html)
cat(tmp_html,file="./htmlTables/htmlTable_multirep.html")



### let's extract full posts with highlighted rep sentences
non_noise_reps = filter(results_bert,Topic!='-1')
extracted_posts = data.table(Topic=c(),Date=c(),Country=c(),Post=c())
for(entry in 1:length(non_noise_reps$Topic)) {
  # remove first and last python array bracket and single quote brackets
  repsentences = str_sub(non_noise_reps[entry,]$Representative_Docs,3,-3)
  repsentences = unlist(str_split(repsentences,"', '"))
  for(sentence in repsentences) {
    thePost = all_posts[unique_sentences[which(unique_sentences$V1==sentence)]$V2+1]
    thePostText = str_replace_all(thePost$V3,pattern=fixed(sentence),replacement=paste("<font color='red'>",sentence,"</font>",sep=""))
    thePostText = str_replace_all(thePostText,pattern="\342\200\231",replacement="'")
    extracted_posts = rbind(extracted_posts,data.table(Topic=c(non_noise_reps[entry,]$Topic),Date=c(thePost$V1),Country=c(thePost$V2),Post=c(thePostText)))
  }
}
extracted_posts=extracted_posts[order(extracted_posts$Topic)]
extracted_posts$Date = paste(as.Date(extracted_posts$Date),"",sep="")

library(xtable)
tmp_html = str_replace_all(pattern=fixed("&lt;/font&gt;"), replacement="</font>",str_replace_all(pattern=fixed("&lt;font color='red'&gt;"),replacement="<font color='red'>",print(xtable(extracted_posts), include.rownames=FALSE,type="html")))
cat(tmp_html,file="./htmlTables/htmlTable_posts.html")


# list topic percentages for manuscript
non_noise_reps = filter(results_bert,Topic!='-1')
abreviated_table = data.table(Topic=c(),Size=c())
for(entry in 1:length(non_noise_reps$Topic)) {
  # remove first and last python array bracket and single quote brackets
  size = paste(round(length(filter(results_all_sentences,Topic==non_noise_reps$Topic[entry])$Topic)/length(results_all_sentences$Topic),4)*100,"%",sep="")
  abreviated_table = rbind(abreviated_table,data.table(Topic=c(non_noise_reps$Topic[entry]),Size=c(size)))
}
