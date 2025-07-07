# Akapulu - Conversational Video Interface

This project is the backend for a conversational video interface, a product of **Akapulu**. We are building a next-generation digital human experience to compete with services like Tavus.

## Project Overview

Our goal is to create a realistic and interactive conversational video interface by leveraging the **NVIDIA digital human blueprint**. This involves integrating various NVIDIA technologies to power real-time rendering, animation, and AI-driven conversation.

## Technology Stack

The core of our infrastructure is built on **Amazon Web Services (AWS)**. For development and deployment, we have access to **EC2 instances** with the following specialized AMIs:

-   **NVIDIA Omniverse Enterprise AMI**: For collaborative 3D workflows and world-building.
-   **NVIDIA Enterprise AMI**: For GPU-accelerated computing and AI model training/inference.

This backend service is responsible for handling the RAG (Retrieval-Augmented Generation) pipeline, which provides the conversational AI with relevant information from a knowledge base of NVIDIA documentation. 