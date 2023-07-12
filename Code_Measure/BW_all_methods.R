##### Comparison of body width data
####################################

library("readxl")
library("stringr")
library("ggplot2")
# 4 sets: Marvin, Sperfeld, Rabus, FullWidth #

Marvin <- read_excel("/home/philipp/Research_Proposal_data/Val_Data_ResearchProp/Test fish bile magna Aig 3.xlsx", sheet = 3)
Sperfeld <- read.csv("~/Data_New_Workflow/Sperfeld_Approach.csv")
Rabus <-read.csv("~/Data_New_Workflow/Rabus_Approach.csv")
FullWidth <- read.csv("~/Data_New_Workflow/FullWidth.csv")
  
## The conversion factor from px to mm
Conv_factor <- 0.002139

### Calculate µm values for all dfs

Sperfeld$Width.µm = Sperfeld$Width.px.*Conv_factor*1000
Rabus$Width.µm = Rabus$Width.px.*Conv_factor*1000
FullWidth$Width.µm = FullWidth$Width.px.*Conv_factor*1000

### To match our dataframes we have to construct and ID
Marvin <- subset(Marvin,Marvin$treatment != "Cf") ### For technical reason the control was doubled
Marvin$ID <- paste(Marvin$treatment,Marvin$repl_NO,Marvin$animal)
Marvin$Method <- "Manual"
MarvinFin <- Marvin[c("ID", "BW", "Method")]
names(MarvinFin)[names(MarvinFin) == 'BW'] <- 'Width.µm'

### For the automatic measurements we have to deconstruct first

### First Sperfeld

Sperfeld[c('temporary', 'animal')] <- str_split_fixed(Sperfeld$image_id, '_', 2)
Sperfeld[c('animal')]<- str_split_fixed(Sperfeld$animal, "[.]", 2)[,1]
Sperfeld$animal <- str_replace(Sperfeld$animal, "0", "")

Sperfeld$treatment <- str_sub(Sperfeld$temporary, end=-2)
Sperfeld$repl_NO <- str_sub(Sperfeld$temporary, - 1, - 1) 
Sperfeld$ID <- paste(Sperfeld$treatment, Sperfeld$repl_NO,Sperfeld$animal)
Sperfeld$Method <- "Sperfeld"
SperfeldFin <- Sperfeld[c('ID', 'Width.µm', "Method")]

### Now Rabus

Rabus[c('temporary', 'animal')] <- str_split_fixed(Rabus$image_id, '_', 2)
Rabus[c('animal')]<- str_split_fixed(Rabus$animal, "[.]", 2)[,1]
Rabus$animal <- str_replace(Rabus$animal, "0", "")

Rabus$treatment <- str_sub(Rabus$temporary, end=-2)
Rabus$repl_NO <- str_sub(Rabus$temporary, - 1, - 1) 
Rabus$ID <- paste(Rabus$treatment, Rabus$repl_NO,Rabus$animal)
Rabus$Method <- "Rabus"
RabusFin <- Rabus[c('ID', 'Width.µm', 'Method')]

## And FullWidth

FullWidth[c('temporary', 'animal')] <- str_split_fixed(FullWidth$image_id, '_', 2)
FullWidth[c('animal')]<- str_split_fixed(FullWidth$animal, "[.]", 2)[,1]
FullWidth$animal <- str_replace(FullWidth$animal, "0", "")

FullWidth$treatment <- str_sub(FullWidth$temporary, end=-2)
FullWidth$repl_NO <- str_sub(FullWidth$temporary, - 1, - 1) 
FullWidth$ID <- paste(FullWidth$treatment, FullWidth$repl_NO,FullWidth$animal)
FullWidth$Method <- "FullWidth"
FullWidthFin <- FullWidth[c('ID', 'Width.µm', "Method")]

### Rename coloumns and Merge
### Also we drop unrealistic low values before hand > 1000µm
RabusFin <- RabusFin[RabusFin$Width.µm > 1000,]
SperfeldFin <- SperfeldFin[SperfeldFin$Width.µm > 1000,]
FullWidthFin <- FullWidthFin[FullWidthFin$Width.µm > 1000,]
AllWidths <- rbind(RabusFin, SperfeldFin, MarvinFin, FullWidthFin)


### Get a clean Number of values: AKA drop all values with comments in Manual 
### Also 4 values in Excel data are not in our Image_Folder therefore we reduce 
### the count in our data from 255 to 251
MarvinNoComm <- subset(MarvinFin, (is.na(Marvin$comment) == TRUE))

AllWidthsDiscard <- subset(AllWidths, (AllWidths$ID %in% MarvinNoComm$ID))

### No we can plot them 
length(AllWidthsDiscard$ID)
length(AllWidths$ID)

## -4 for non existing images
AllWidthsDiscard %>%
  ggplot(aes(Method, Width.µm)) +
  geom_violin(aes(fill=Method)) +
  geom_point(width = 0.2, height = 1.5)+
  #geom_line(aes(group = ID), col = "light grey", lty = "dashed") + 
  theme_minimal()+ theme(text = element_text(size = 20)) +
  annotate(geom="text", x=1, y=1200, label= paste("n =",length(FullWidthFin$Width.µm),"/",length(MarvinFin$Width.µm)-4))+
  annotate(geom="text", x=2, y=1200, label= paste("n =",length(MarvinFin$Width.µm)-4,"/",length(MarvinFin$Width.µm)-4))+
  annotate(geom="text", x=3, y=1200, label= paste("n =",length(RabusFin$Width.µm),"/",length(MarvinFin$Width.µm)-4))+
  annotate(geom="text", x=4, y=1200, label= paste("n =",length(SperfeldFin$Width.µm),"/",length(MarvinFin$Width.µm)-4))+
  ylab("Body width [µm]") + ggtitle("Body width comparison | Marvin validation")





