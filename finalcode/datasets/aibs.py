"""AIBS/PeerRead peer review dataset loader for science domain"""
import pandas as pd
import numpy as np
import json
import os
from pathlib import Path
from typing import List, Dict, Any


class AIBSDataset:
    """
    Loader for peer review datasets (PeerRead or AIBS format)
    
    Contains scientific paper abstracts and review decisions,
    useful for paper quality assessment and review allocation.
    Supports both real PeerRead data and synthetic fallback.
    """
    
    def __init__(self, data_dir: str):
        """
        Args:
            data_dir: Path to data directory (tries PeerRead, AIBS, then synthetic)
        """
        self.data_dir = Path(data_dir)
        self.papers = None
        self.reviews = None
        self.use_synthetic = False
        self.data_source = None
    
    def load(self):
        """Load peer review dataset"""
        print("Loading peer review dataset...")
        
        # Try PeerRead format first
        peerread_dir = self.data_dir.parent / "peerread"
        if peerread_dir.exists():
            try:
                self.papers, self.reviews = self._load_peerread(peerread_dir)
                self.data_source = "PeerRead"
                print(f"✓ Loaded {len(self.papers)} papers from PeerRead dataset")
                return self
            except Exception as e:
                print(f"Warning: Failed to load PeerRead data ({e})")
        
        # Try AIBS format
        paper_file = self.data_dir / "papers.csv"
        review_file = self.data_dir / "reviews.csv"
        
        if paper_file.exists() and review_file.exists():
            self.papers = pd.read_csv(paper_file)
            self.reviews = pd.read_csv(review_file)
            self.data_source = "AIBS"
            print(f"✓ Loaded {len(self.papers)} papers from AIBS dataset")
        else:
            print("Warning: No real data found. Using synthetic data.")
            self.papers, self.reviews = self._generate_synthetic_data()
            self.use_synthetic = True
            self.data_source = "Synthetic"
        
        return self
    
    def _load_peerread(self, peerread_dir: Path) -> tuple:
        """Load PeerRead dataset from JSON files"""
        data_dir = peerread_dir / "data"
        if not data_dir.exists():
            # Try if peerread_dir itself contains venue folders
            data_dir = peerread_dir
        
        papers_data = []
        reviews_data = []
        
        # Find all JSON files recursively
        json_files = list(data_dir.rglob("*.json"))
        print(f"Found {len(json_files)} JSON files in PeerRead")
        
        if len(json_files) == 0:
            raise ValueError("No JSON files found in PeerRead directory")
        
        paper_id = 0
        for json_file in json_files[:500]:  # Limit to 500 papers for efficiency
            try:
                with open(json_file, 'r', encoding='utf-8', errors='ignore') as f:
                    paper_data = json.load(f)
                
                # Extract paper info
                title = paper_data.get('title', 'Unknown Title')
                abstract = paper_data.get('abstract', '')
                
                # Extract venue from path
                venue = self._extract_venue(str(json_file))
                
                # Add paper
                papers_data.append({
                    'paper_id': paper_id,
                    'title': title,
                    'abstract': abstract,
                    'domain': venue,
                    'submission_year': 2017  # Default year
                })
                
                # Extract reviews
                if 'reviews' in paper_data and isinstance(paper_data['reviews'], list):
                    for rev_idx, review in enumerate(paper_data['reviews']):
                        if isinstance(review, dict):
                            # Extract rating/score
                            score = self._extract_score(review)
                            decision = self._extract_decision(review, score)
                            
                            reviews_data.append({
                                'paper_id': paper_id,
                                'reviewer_id': f"{paper_id}_{rev_idx}",
                                'score': score,
                                'decision': decision
                            })
                
                paper_id += 1
                
            except Exception as e:
                continue  # Skip problematic files
        
        if len(papers_data) == 0:
            raise ValueError("No valid papers extracted from PeerRead")
        
        papers_df = pd.DataFrame(papers_data)
        reviews_df = pd.DataFrame(reviews_data)
        
        print(f"Parsed {len(papers_df)} papers with {len(reviews_df)} reviews")
        return papers_df, reviews_df
    
    def _extract_venue(self, file_path: str) -> str:
        """Extract venue from file path"""
        path_lower = file_path.lower()
        if 'nips' in path_lower or 'neurips' in path_lower:
            return 'neuroscience'
        elif 'acl' in path_lower or 'conll' in path_lower:
            return 'computer_science'
        elif 'iclr' in path_lower:
            return 'machine_learning'
        else:
            return 'general'
    
    def _extract_score(self, review: Dict) -> int:
        """Extract numerical score from review"""
        # Try common score field names
        for field in ['rating', 'recommendation', 'overall', 'score', 'RECOMMENDATION']:
            if field in review:
                try:
                    val = review[field]
                    # Handle numeric values
                    if isinstance(val, (int, float)):
                        return int(val)
                    # Handle string values like "8: Accept"
                    if isinstance(val, str):
                        # Extract first number
                        import re
                        match = re.search(r'\d+', val)
                        if match:
                            return int(match.group())
                except:
                    pass
        
        # Default score
        return np.random.randint(1, 11)
    
    def _extract_decision(self, review: Dict, score: int) -> str:
        """Extract decision from review"""
        # Try explicit decision fields
        for field in ['decision', 'recommendation', 'RECOMMENDATION']:
            if field in review:
                decision_str = str(review[field]).lower()
                if 'accept' in decision_str:
                    return 'accept'
                elif 'reject' in decision_str:
                    return 'reject'
                elif 'revise' in decision_str or 'revision' in decision_str:
                    return 'revise'
        
        # Infer from score
        if score >= 7:
            return 'accept'
        elif score <= 4:
            return 'reject'
        else:
            return 'revise'
    
    def _generate_synthetic_data(self, num_papers: int = 500) -> tuple:
        """Generate synthetic peer review data"""
        np.random.seed(43)
        
        papers = pd.DataFrame({
            'paper_id': range(num_papers),
            'title': [f"Paper {i}: Scientific Study" for i in range(num_papers)],
            'abstract': [f"Abstract content for paper {i}..." for i in range(num_papers)],
            'domain': np.random.choice(['biology', 'neuroscience', 'medicine'], num_papers),
            'submission_year': np.random.randint(2018, 2024, num_papers)
        })
        
        reviews = pd.DataFrame({
            'paper_id': np.random.choice(range(num_papers), 1500),
            'reviewer_id': np.random.randint(0, 100, 1500),
            'score': np.random.randint(1, 11, 1500),
            'decision': np.random.choice(['accept', 'reject', 'revise'], 1500)
        })
        
        return papers, reviews
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics"""
        stats = {
            'data_source': self.data_source,
            'num_papers': len(self.papers),
            'num_reviews': len(self.reviews),
            'use_synthetic': self.use_synthetic
        }
        
        if self.reviews is not None and len(self.reviews) > 0:
            # Review statistics
            stats['avg_reviews_per_paper'] = len(self.reviews) / len(self.papers)
            
            if 'decision' in self.reviews.columns:
                decision_counts = self.reviews['decision'].value_counts()
                total = len(self.reviews)
                stats['accept_rate'] = decision_counts.get('accept', 0) / total if total > 0 else 0
                stats['reject_rate'] = decision_counts.get('reject', 0) / total if total > 0 else 0
                stats['revise_rate'] = decision_counts.get('revise', 0) / total if total > 0 else 0
            
            if 'score' in self.reviews.columns:
                stats['avg_score'] = self.reviews['score'].mean()
                stats['std_score'] = self.reviews['score'].std()
        
        if self.papers is not None and 'domain' in self.papers.columns:
            domain_counts = self.papers['domain'].value_counts()
            stats['domains'] = domain_counts.to_dict()
        
        return stats
    
    def get_paper(self, paper_id: int) -> Dict:
        """Get paper details"""
        paper = self.papers[self.papers['paper_id'] == paper_id].iloc[0]
        
        return {
            'paper_id': int(paper['paper_id']),
            'title': paper['title'],
            'abstract': paper['abstract'],
            'domain': paper['domain']
        }
    
    def get_task_batch(self, batch_size: int = 32) -> List[Dict]:
        """
        Get a batch of paper review tasks
        
        Args:
            batch_size: Number of tasks
            
        Returns:
            List of review task dictionaries
        """
        sampled_papers = self.papers.sample(min(batch_size, len(self.papers)))
        
        tasks = []
        for _, paper in sampled_papers.iterrows():
            task = {
                'domain': 'science',
                'task_type': 'paper_review',
                'paper_id': int(paper['paper_id']),
                'title': paper['title'],
                'abstract': paper['abstract'],
                'query': f"Assess the quality and provide review for: {paper['title']}"
            }
            tasks.append(task)
        
        return tasks
    
    def get_requirement_vector(self) -> np.ndarray:
        """
        Get typical requirement vector for science tasks
        
        Returns:
            5-dimensional capability requirement
        """
        # Science domain emphasizes: reasoning, knowledge, analysis, generation
        return np.array([0.9, 0.8, 0.9, 0.6, 0.7])
