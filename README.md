# Microservice composition evolution suported by Machine Learning

This repositry contains the artifacts developed to integrate machine learning techniques into the [software infrastructure](https://github.com/microserviceresearch/microservices-composition-infrastructure) developed to support the composition of microservices.

You can find here the following:

* CaseStudies: four case studies modelled in BPMN that have been used to manually test de application of a [catalogue of adaptation rules](https://microserviceresearch.github.io/AdaptationRuleCatalogue.pdf).
* Dataset: a folder with the csv files used to the R script
* MicroserviceScript.R: An script implemented in the R language in order to train different machine learning techniques from the previous dataset


# Description

Microservices need to be composed to support the business processes of organizations. With the aim of maintaining a lower coupling among micro-services, these compositions are usually implemented by means of event-based choreographies. This increases the independence among microservices for deployment and evolution, which is one of the most important goals pursued by this type of architecture. However, choreographies split the control flow of compositions among the different participant microservices, which makes them hard to analyse and understand when requirements change, and the composition needs to globally evolve. 

We have developed an approach based on the choreography of BPMN fragments. According to this approach, business process engineers create the big picture of the microservice composition through a BPMN model. Then, this model is split into BPMN fragments which are executed through an event-based choreography. This composition approach is supported by a [microservice architecture](https://github.com/microserviceresresearch/microservices-composition-infrastructure) developed to achieve that both descriptions of a composition, the big picture and the split one, coexist in the same system.

One of the main problems to be solved by choreographies is related to their evolution due to the complexity that the need of integrating changes among autonomous and independent partners introduces. In our microservice composition approach, a change introduced from the local perspective of a microservice needs to be integrated with both the BPMN fragment of the other partners and the big picture of the composition. 

To face this challenge we have developed a solution based on machine learning techniques. This repository contains the sotware artifacts used to develop this solution 
