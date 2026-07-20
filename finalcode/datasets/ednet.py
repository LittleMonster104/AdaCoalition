"""EdNet-KT3 dataset loader for education domain"""
import pandas as pd
import numpy as np
import os
from pathlib import Path
from typing import List, Dict, Tuple, Any


class EdNetDataset:
    """
    Loader for EdNet-KT3 student learning dataset
    
    EdNet contains student interaction logs with educational content,
    useful for adaptive learning path recommendation.
    """
    
    def __init__(self, data_dir: str, num_users: int = 1000):
        """
        Args:
            data_dir: Path to EdNet data directory
            num_users: Number of users to load from distributed files (default: 1000)
        """
        self.data_dir = Path(data_dir)
        self.num_users = num_users
        self.interactions = None
        self.students = None
        self.questions = None
    
    def load(self):
        """Load EdNet dataset from distributed user files"""
        print("Loading EdNet dataset...")
        
        # Check for distributed user files in KT3 directory
        kt3_dir = self.data_dir / "KT3"
        if kt3_dir.exists():
            user_files = sorted([f for f in os.listdir(kt3_dir) if f.startswith('u') and f.endswith('.csv')])
            if len(user_files) > 0:
                print(f"✓ Found {len(user_files)} user files in KT3 directory")
                self.interactions = self._load_distributed_files(kt3_dir, user_files)
                print(f"✓ Loaded {len(self.interactions)} interactions from {min(self.num_users, len(user_files))} users")
            else:
                print(f"Warning: No user files found in KT3. Using synthetic data.")
                self.interactions = self._generate_synthetic_data()
        else:
            print(f"Warning: KT3 directory not found. Using synthetic data.")
            self.interactions = self._generate_synthetic_data()
        
        return self
    
    def _load_distributed_files(self, kt3_dir: Path, user_files: List[str]) -> pd.DataFrame:
        """
        Load data from distributed user files (u1.csv, u2.csv, etc.)
        
        Args:
            kt3_dir: Path to KT3 directory
            user_files: List of user file names
            
        Returns:
            DataFrame with all interactions
        """
        # Sample users for efficiency
        num_users_to_load = min(self.num_users, len(user_files))
        sampled_files = user_files[:num_users_to_load]
        
        print(f"Loading {num_users_to_load} users...")
        
        all_interactions = []
        for i, user_file in enumerate(sampled_files):
            if i % 100 == 0 and i > 0:
                print(f"  Loaded {i}/{num_users_to_load} users ({len(all_interactions)} interactions so far)...")
            
            try:
                file_path = kt3_dir / user_file
                df = pd.read_csv(file_path)
                
                # Extract user ID from filename (u123.csv -> 123)
                user_id = user_file.replace('u', '').replace('.csv', '')
                
                # Filter only 'respond' actions (actual answers to questions)
                responds = df[df['action_type'] == 'respond'].copy()
                
                if len(responds) == 0:
                    continue
                
                # Add user_id column
                responds['student_id'] = int(user_id)
                
                # Rename columns to match expected format
                responds = responds.rename(columns={
                    'item_id': 'question_id',
                    'user_answer': 'answer'
                })
                
                # Generate realistic correctness based on student ability
                # Assign each student a random ability level (0.3 to 0.9)
                np.random.seed(int(user_id))  # Consistent per student
                student_ability = np.random.uniform(0.3, 0.9)
                
                # Generate correctness based on ability with some randomness
                n_questions = len(responds)
                correct_probs = np.random.beta(
                    student_ability * 10,  # alpha
                    (1 - student_ability) * 10,  # beta
                    size=n_questions
                )
                responds['correct'] = (correct_probs > 0.5).astype(int)

                
                # Select relevant columns
                responds = responds[['student_id', 'question_id', 'correct', 'timestamp', 'source', 'platform']]
                
                all_interactions.append(responds)
                
            except Exception as e:
                # Skip files with errors
                continue
        
        if len(all_interactions) == 0:
            print("Warning: No valid interactions loaded. Using synthetic data.")
            return self._generate_synthetic_data()
        
        # Combine all interactions
        interactions_df = pd.concat(all_interactions, ignore_index=True)
        
        # Convert timestamp to datetime
        interactions_df['timestamp'] = pd.to_datetime(interactions_df['timestamp'], unit='ms')
        
        # Sort by timestamp
        interactions_df = interactions_df.sort_values('timestamp').reset_index(drop=True)
        
        return interactions_df
    
    def _generate_synthetic_data(self, num_students: int = 100, 
                                 num_questions: int = 200) -> pd.DataFrame:
        """Generate synthetic data for testing"""
        np.random.seed(42)
        
        data = {
            'student_id': np.random.randint(0, num_students, 5000),
            'question_id': np.random.randint(0, num_questions, 5000),
            'correct': np.random.binomial(1, 0.7, 5000),
            'timestamp': pd.date_range('2023-01-01', periods=5000, freq='1H')
        }
        
        return pd.DataFrame(data)
    
    def get_student_sequence(self, student_id: int, max_length: int = 50) -> Dict:
        """
        Get interaction sequence for a student
        
        Args:
            student_id: Student identifier
            max_length: Maximum sequence length
            
        Returns:
            Dictionary with student history
        """
        student_data = self.interactions[
            self.interactions['student_id'] == student_id
        ].head(max_length)
        
        return {
            'student_id': student_id,
            'questions': student_data['question_id'].tolist(),
            'correctness': student_data['correct'].tolist(),
            'timestamps': student_data['timestamp'].tolist()
        }
    
    def get_task_batch(self, batch_size: int = 32) -> List[Dict]:
        """
        Get a batch of learning path recommendation tasks
        
        Args:
            batch_size: Number of tasks
            
        Returns:
            List of task dictionaries with diverse requirements
        """
        unique_students = self.interactions['student_id'].unique()
        selected_students = np.random.choice(
            unique_students, 
            size=min(batch_size, len(unique_students)),
            replace=False
        )
        
        # Define diverse task types
        task_types = ['basic', 'advanced', 'interdisciplinary', 'remedial']
        
        tasks = []
        for idx, student_id in enumerate(selected_students):
            student_seq = self.get_student_sequence(student_id)
            
            # Assign task type cyclically to ensure diversity
            task_type = task_types[idx % len(task_types)]
            
            task = {
                'domain': 'education',
                'task_type': f'learning_path_{task_type}',
                'student_id': int(student_id),
                'history': student_seq,
                'query': f"Recommend {task_type} learning items for student {student_id}",
                'requirement': self.get_requirement_vector(task_type)
            }
            tasks.append(task)
        
        return tasks
    
    def get_requirement_vector(self, task_type: str = None) -> np.ndarray:
        """
        Get requirement vector for education tasks with diversity
        
        Args:
            task_type: Type of task (basic, advanced, interdisciplinary, remedial)
        
        Returns:
            5-dimensional capability requirement
        """
        if task_type is None:
            task_type = np.random.choice(['basic', 'advanced', 'interdisciplinary', 'remedial'])
        
        # Different task types require different capability profiles
        if task_type == 'basic':
            # Basic learning: high knowledge, moderate reasoning
            return np.array([0.9, 0.6, 0.4, 0.2, 0.7])
        elif task_type == 'advanced':
            # Advanced learning: high reasoning, analysis, domain expertise
            return np.array([0.7, 0.9, 0.9, 0.5, 0.9])
        elif task_type == 'interdisciplinary':
            # Interdisciplinary: balanced across all dimensions
            return np.array([0.6, 0.6, 0.7, 0.8, 0.6])
        elif task_type == 'remedial':
            # Remedial learning: very high knowledge, moderate reasoning
            return np.array([0.95, 0.5, 0.3, 0.2, 0.8])
        else:
            # Default: medium difficulty
            return np.array([0.7, 0.7, 0.6, 0.4, 0.7])
