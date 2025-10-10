import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class ResponseFormatter:
    
    @staticmethod
    def generate_summary(
        violence_result: Optional[Dict[str, Any]], 
        weapon_result: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive summary from violence and weapon detection results
        
        Args:
            violence_result: Results from violence detection
            weapon_result: Results from weapon detection
            
        Returns:
            Summary dictionary for frontend display (no threat/risk levels)
        """
        summary = {
            "detections_found": False,
            "violence_detected": False,
            "weapons_detected": False,
            "analysis_details": {},
            "confidence_scores": {},
            "processing_stats": {},
            "analysis_timestamp": datetime.now().isoformat()
        }
        
        # Analyze violence detection results
        if violence_result:
            summary["violence_detected"] = violence_result.get("is_violence", False)
            if summary["violence_detected"]:
                summary["detections_found"] = True
                summary["analysis_details"]["violence"] = {
                    "detected": True,
                    "confidence": violence_result.get("confidence", 0.0),
                    "description": "Violence patterns detected in video"
                }
                summary["confidence_scores"]["violence"] = violence_result.get("confidence", 0.0)
        
        # Analyze weapon detection results  
        if weapon_result:
            summary["weapons_detected"] = weapon_result.get("weapons_detected", False)
            if summary["weapons_detected"]:
                summary["detections_found"] = True
                
                # Get confidence scores from new format
                confidence_scores = weapon_result.get("confidence_scores", {})
                detected_weapons = weapon_result.get("detected_weapons", [])
                
                summary["analysis_details"]["weapons"] = {
                    "detected": True,
                    "weapons_found": detected_weapons,
                    "confidence_scores": confidence_scores,
                    "description": f"Detected {len(detected_weapons)} weapon type(s)"
                }
                
                # Add individual weapon confidence scores
                for weapon, details in confidence_scores.items():
                    summary["confidence_scores"][f"weapon_{weapon}"] = details.get("confidence", 0.0)
                
                # Add weapon details to analysis
                if detected_weapons:
                    weapon_descriptions = []
                    for weapon in detected_weapons:
                        if weapon in confidence_scores:
                            conf_details = confidence_scores[weapon]
                            weapon_descriptions.append(
                                f"{weapon.title()}: {conf_details.get('confidence', 0.0):.3f} "
                                f"({conf_details.get('confidence_level', 'Unknown')})"
                            )
                    
                    if weapon_descriptions:
                        summary["analysis_details"]["weapons"]["detailed_results"] = weapon_descriptions
        
        # Add processing statistics if available
        if violence_result and "processing_stats" in violence_result:
            summary["processing_stats"]["violence"] = violence_result["processing_stats"]
        
        if weapon_result and "processing_stats" in weapon_result:
            summary["processing_stats"]["weapons"] = weapon_result["processing_stats"]
        
        logger.info(f"📋 Generated summary: Violence={summary['violence_detected']}, Weapons={summary['weapons_detected']}")
        return summary
    
    
    @staticmethod
    def format_detection_results(
        violence_result: Optional[Dict[str, Any]], 
        weapon_result: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Format detection results for API response
        
        Args:
            violence_result: Results from violence detection
            weapon_result: Results from weapon detection
            
        Returns:
            Formatted results dictionary
        """
        formatted = {
            "violence_analysis": None,
            "weapon_analysis": None,
            "summary": ResponseFormatter.generate_summary(violence_result, weapon_result)
        }
        
        # Format violence results
        if violence_result:
            formatted["violence_analysis"] = {
                "is_violence": violence_result.get("is_violence", False),
                "confidence": violence_result.get("confidence", 0.0),
                "analysis_details": violence_result.get("analysis", {}),
                "processing_stats": violence_result.get("processing_stats", {})
            }
        
        # Format weapon results
        if weapon_result:
            formatted["weapon_analysis"] = {
                "weapons_detected": weapon_result.get("weapons_detected", False),
                "detected_weapons": weapon_result.get("detected_weapons", []),
                "confidence_scores": weapon_result.get("confidence_scores", {}),
                "processing_stats": weapon_result.get("processing_stats", {}),
                "analysis_summary": weapon_result.get("analysis_summary", {})
            }
        
        return formatted