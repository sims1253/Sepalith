"""Pinned CPU parser bootstrap; synthetic bytes only, no corpus discovery."""
import hashlib,importlib.util,json,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
INTEGRATION=HERE.parent/'final-production-integration-v3/production_integration_v3.py'
def load_integration():
    spec=importlib.util.spec_from_file_location('dat08_native_final_integration',INTEGRATION)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    module._closure()
    return module

def preload():
    integration=load_integration()
    modules={key:integration._load(name,path) for key,name,path in (
        ('guard','dat08_integration_guard',integration.GUARD_PATH),
        ('builder','dat08_integration_builder',integration.BUILDER_PATH),
        ('families','dat08_integration_raw_families_v3',integration.RAW_FAMILIES_V3_PATH),
        ('finish','dat08_integration_finish_v4',integration.FINISH_V4_PATH),
        ('adapter','dat08_integration_evaluator_adapter_v3',integration.ADAPTER_PATH))}
    fixtures=[
        ('rename_propagation',b'f <- function(x) {\n  target <- x\n  print(target)\n  y <- target\n  target\n}\n'),
        ('pipe_rewrite',b'f <- function(x) {\n  a <- x %>% foo()\n  b <- x %>% bar()\n}\n'),
        ('na_rm_propagation',b'f <- function(x) {\n  dplyr::summarise(x, mean(x))\n  dplyr::summarise(x, sd(x))\n}\n'),
        ('format_propagation',b'f<-function(x){\na<-1\nb<-2\n}\ng<-function(y){\nc<-3\nd<-4\n}\n'),
        ('no_op',b'f <- function(x) {\n  x + 1\n}\nvalue <- 1\n'),
        ('finish_block',b"#' Add values.\n#' @param x Numeric values.\n#' @return Numeric values.\nadd_values <- function(x) {\n  first <- x + 1\n  second <- first * 2\n  c(first, second)\n}")]
    results=[]
    for family,raw in fixtures:
        kwargs={'package_id':'synthetic-package','path':'R/synthetic.R','group_id':'synthetic-group','family':family,'source_sha256':hashlib.sha256(raw).hexdigest()}
        if family=='format_propagation':
            normalized=b'f <- function(x) {\n  a <- 1\n  b <- 2\n}\ng <- function(y) {\n  c <- 3\n  d <- 4\n}\n'
            kwargs.update(normalized_bytes=normalized,normalized_path='R/synthetic.R',normalized_sha256=hashlib.sha256(normalized).hexdigest(),seed=0)
        if family=='finish_block':result=modules['finish'].build_raw_family_case(raw,**kwargs,variant='signature')
        elif family=='no_op':result=modules['families'].build_raw_family_case(raw,**kwargs)
        else:result=modules['builder'].build_raw_source_case(raw,**kwargs)
        results.append({'family':family,'status':result['status'],'source_sha256':kwargs['source_sha256']})
    modules['adapter']._load_protocol()
    return integration,results
