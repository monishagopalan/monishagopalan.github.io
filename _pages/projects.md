---
layout: single
read_time: false
comments: false
share: false
title: Projects
permalink: /projects/
custom_css:
  - /assets/css/projects.css
custom_js:
  - /assets/js/projects.js
classes: wide
---
<!-- New Project -->
<div class="project-container">
  <!-- Header Section -->
  <div class="project-header-section">
    <h2 class="project-main-title">Hopfield Network for Asset Allocation</h2>
    <div class="project-links">
      <a href="/assets/documents/hopfield-networks.pdf" target="_blank" title="Read Paper"><i class="fas fa-download"></i></a>
      <a href="https://doi.org/10.1145/3677052.3698605">
        <img src="https://img.shields.io/badge/ACM_ICAIF--24-10.1145/3677052.3698605-blue" alt="ACM ICAIF-24 Paper">
      </a>
    </div>
  </div>

  <!-- 2x2 Grid for Details -->
  <div class="project-details-grid">
    
    <!-- Card 1: Problem -->
    <div class="detail-card">
      <div class="card-title">The Problem</div>
      <div class="card-content">
        <p> Standard portfolio optimization often suffers from estimation errors, leading to financial losses. While deep learning offers better pattern recognition, existing models like <b>LSTMs</b> are computationally heavy and complex, creating a need for faster, more stable alternatives. </p>
      </div>
    </div>

    <!-- Card 2: Solution -->
    <div class="detail-card">
      <div class="card-title">Solution</div>
      <div class="card-content">
        <p> We implemented Modern Hopfield Networks to capture complex market patterns efficiently. By replacing standard layers with Hopfield layers, this solution matches or beats state-of-the-art models like Transformers while being significantly faster and more stable to train. </p>
      </div>
    </div>

    <!-- Card 3: Skills -->
    <div class="detail-card">
      <div class="card-title">Skills Developed</div>
      <div class="card-content">
        <p> Mastered advanced Deep Learning architectures: <b>Hopfield Networks</b>, <b>Transformers</b> and time-series embedding: <b>Time2Vec</b>. Gained expertise in quantitative finance metrics: <b>Sharpe/Sortino ratios</b> and rigorous backtesting strategies: <b>Combinatorial Purged Cross-Validation.</b></p>
      </div>
    </div>

    <!-- Card 4: Applications -->
    <div class="detail-card">
      <div class="card-title">Applications</div>
      <div class="card-content">
        <p> This project offers a scalable tool for dynamic asset allocation and risk management. The architecture is flexible enough to incorporate unconventional data sources, such as ESG ratings or market sentiment, to enhance investment decision-making.</p>
      </div>
    </div>

  </div>
</div>

<div class="project-container">
  <!-- Header Section -->
  <div class="project-header-section">
    <h2 class="project-main-title">Corporate Credit Rating Forecast using Machine Learning Methods</h2>
    <div class="project-links">
      <a href="https://github.com/monishagopalan/credit-rating-forecast">
        <img src="https://img.shields.io/badge/GitHub-View_Repository-blue?logo=GitHub" alt="GitHub Repository">
      </a>
      <a href="https://monishagopalan.github.io/projects/credit-rating/">
        <img src="https://img.shields.io/badge/Blog-Read%20Now-brightgreen" alt="Blog Post">
      </a>
    </div>
  </div>

  <!-- 2x2 Grid for Details -->
  <div class="project-details-grid">
    
    <!-- Card 1: Problem -->
    <div class="detail-card">
      <div class="card-title">The Problem</div>
      <div class="card-content">
        <p>Corporate credit ratings, issued by credit rating agencies like Standard & Poor's and Moody's, express the agency's opinion about the ability of a company to meet its debt obligations. Each agency applies its own methodology to measure creditworthiness and this assessment is an expensive and complicated process. Usually, the agencies take time to provide new ratings and update older ones. This causes delays in decision-making process for investors who use these ratings to assess their credit risk.</p>
      </div>
    </div>

    <!-- Card 2: Solution -->
    <div class="detail-card">
      <div class="card-title">Solution</div>
      <div class="card-content">
        <p>One solution to address delays would be to use the historical financial information of a company to build a predictive quantitative model capable of forecasting the credit rating that a company will receive. I employed machine learning techniques, creating classification models that quickly forecast credit ratings.</p>
      </div>
    </div>

    <!-- Card 3: Skills -->
    <div class="detail-card">
      <div class="card-title">Skills Developed</div>
      <div class="card-content">
        <p>Explored classification methods like <b>XGBoost</b>, <b>RandomForest</b> and techniques to address imbalance in datasets - <a href="{% post_url 2024-01-01-smote %}">SMOTE</a>. Also delved into <b>financial ratios</b> gaining knowledge on understanding a company's fiscal strength.</p>
      </div>
    </div>

    <!-- Card 4: Applications -->
    <div class="detail-card">
      <div class="card-title">Applications</div>
      <div class="card-content">
        <p>The insights gained can aid financial analysts, investors, and companies in making more informed and quick decisions related to credit risk. The classification methods used here can also be used to forecast other ratings like ESG Ratings.</p>
      </div>
    </div>

  </div>
</div>

