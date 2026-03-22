import { useNavigate } from 'react-router-dom';
import cuffs from '../assets/cuffs.png'
import fingerprint from '../assets/fingerprint.png'
import cctv from '../assets/cctv.jpg'
import dataset from '../assets/dataset.png'
import preprocessing from '../assets/preprocessing.png'
import model from '../assets/model.png'

export default function LandingPage() {
  const navigate = useNavigate();

  return (
    <div className="landing-page">
      {/* HOME */}
      <section id="home" className="hero-section">
        <div className="hero-deco hero-deco-left">
          <img src={cuffs} alt="handcuffs" className='deco-svg' />
        </div>
        <div className="hero-deco hero-deco-right">
          <img src={fingerprint} alt="handcuffs" className='deco-svg' />
        </div>

        <div className="hero-content">
          <h1 className="hero-title">
            ANALYSIS OF{' '}
            <span className="text-gold">SELF-SUPERVISED LEARNING (SSL)</span>{' '}
            FRAMEWORKS IN COMPUTER VISION FOR{' '}
            <span className="text-gold">SUSPICIOUS BEHAVIOR DETECTION</span>
          </h1>
          <p className="hero-subtitle">
            Presented to the Faculty of the{' '}
            <strong>Institute of Computer Science, University of the Philippines Los Baños</strong>{' '}
            in partial fulfillment of the requirements for the Degree of{' '}
            <strong>Bachelor of Science in Computer Science</strong>
          </p>
        </div>
      </section>

      {/* ABOUT THE STUDY */}
      <section id="about" className="about-section">
        <div className="about-content">
          <div className="about-text">
            <div className="section-badge">Research Problem and Objectives</div>
            <p>
              Crowded public spaces in the Philippines require more efficient security measures 
              to address growing public safety threats. Although CCTV systems are widely used, 
              surveillance still heavily depends on security personnel, making real-time monitoring 
              physically demanding and prone to human error.
            </p>
            <p>
              Despite the advances in AI and Computer Vision for surveillance systems, detecting suspicious 
              behaviors in real-time videos remains a challenge, as traditional supervised learning models 
              require manually annotated video footage to learn how to recognize specific actions. This need 
              for manual labeling of data makes developing a security system expensive and time-consuming to train.

            </p>
            <p>
              This study aims to develop a computer system that uses Computer Vision Self-Supervised Learning (SSL) 
              frameworks to detect suspicious behaviors in surveillance videos. The system is designed 
              to reduce dependence on manually labeled data, making training more efficient and scalable.
            </p>
            <p>
              The study utilizes the UCF Crime Dataset and compares SSL frameworks 
              with CNN and Vision Transformer (ViT) backbones in terms of detection 
              accuracy and computational efficiency.
            </p>
          </div>
          <div className="about-image">
            <img src={cctv} alt="CCTV surveillance camera" />
          </div>
        </div>
      </section>

      {/* METHODS */}
      <section id="methods" className="methods-section">
        <h2 className="methods-title">Methodology</h2>
        <div className="methods-cards">
          <div className="method-card">
            <div className="method-icon">
              <img src={dataset} alt="dataset-icon" />
            </div>
            <p className="method-description">
              This study utilizes the UCF-Crime surveillance video dataset, 
              focusing specifically on Robbery, Assault, Shooting, Abuse, and a Normal baseline. 
              This targeted selection addresses computational resources issues and
               directly addresses the most prevalent crimes in the Philippines.
            </p>
            <span className="method-label">Dataset</span>
          </div>

          <div className="method-card">
            <div className="method-icon">
              <img src={preprocessing} alt="data-preprocessing" />
            </div>
            <p className="method-description">
              To optimize the model's performance, raw videos are filtered to extract only motion-relevant frames.
              These key frames then undergo data augmentation, such as random resized cropping, horizontal flipping, 
              and color jittering, to maximize dataset variability.
            </p>
            <span className="method-label">Data Preprocessing</span>
          </div>

          <div className="method-card">
            <div className="method-icon">
              <img src={model} alt="model-icon" />
            </div>
            <p className="method-description">
              This study implements and compares two Self-Supervised Learning (SSL) frameworks: 
              the CNN-based BYOL (ResNet-50) and the transformer-based DINOv3 (ViT). 
              This setup allows for a direct performance comparison between these architectures 
              for detecting suspicious behavior in raw surveillance footage.
            </p>
            <span className="method-label">Frameworks</span>
          </div>
        </div>

        <button className="methods-cta" onClick={() => navigate('/analyze')}>
          Try SSL Models →
        </button>
      </section>
    </div>
  );
}
