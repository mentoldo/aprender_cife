## Creamos la carpeta con los datos en el caso que no exista
if (!dir.exists('data')) {
    dir.create('data')
    
    ## Descargamos las bases de 2017
    files <- list(c("primaria", "https://www.argentina.gob.ar/sites/default/files/aprender2017-primaria-6.zip"),
                  c("secundaria", "https://www.argentina.gob.ar/sites/default/files/aprender2017-secundaria-12.zip")
    )
    
    lapply(files, function(x){download.file(x[2], paste("./data/", x[1], ".zip", sep = ""))})
    
    lapply(list.files("data/", "*.zip"), unzip)
}


## Descargamos archivos de documentación
if (!dir.exists('docs')) {
    dir.create('docs')
    
    ## Descargamos los diccionarios
    files <- list(c("dict_primaria", "https://www.argentina.gob.ar/sites/default/files/aprender2017-diccionario-primaria-6.xlsx"),
                  c("dicc_secundaria", "https://www.argentina.gob.ar/sites/default/files/aprender2017-diccionario-secundaria-12_0.xlsx")
    )
    
    lapply(files, function(x){download.file(x[2], paste("./docs/", x[1], ".xlsx", sep = ""))})
    
    ## Descargamos los cuestionarios
    files <- list(c("dict_primaria", "https://www.argentina.gob.ar/sites/default/files/cuadernillo_estudiante_primaria.pdf"),
                  c("dicc_secundaria", "https://www.argentina.gob.ar/sites/default/files/cuadernillo_estudiante_secundaria.pdf")
    )
    
    lapply(files, function(x){download.file(x[2], paste("./docs/", x[1], ".pdf", sep = ""))})
}
    