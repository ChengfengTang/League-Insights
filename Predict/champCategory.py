"""
League of Legends Champion Jungler Categorization
This module categorizes champions by their jungler playstyle for prediction purposes.
Categories are used to train separate models or apply category-specific logic.
"""

# Champion categorization by jungler playstyle
CHAMPION_CATEGORIES = {
    # Aggressive - Early game gankers, strong early pressure
    "aggressive": [
        "LeeSin",
        "JarvanIV",
        "Elise",
        "RekSai",
        "Nidalee",
        "XinZhao",
        "Vi",
        "Pantheon",
        "Sett",
        "Warwick",
        "Olaf",
        "Graves",
        "Kindred",
        "KhaZix",
        "Rengar",
        "Shaco",
        "Evelynn",
        "Talon",
        "Qiyana",
        "Zed",
        "Nocturne",
        "Taliyah",
        "Gragas",
        "Sejuani",
        "Zac",
        "Hecarim",
        "Udyr",
        "Volibear",
        "Trundle",
        "Diana",
        "Ekko",
        "Fizz",
        "Camille",
    ],
    
    # Full Clear - Champions that prioritize full clearing camps before ganking
    "full_clear": [
        "Fiddlesticks",
        "Karthus",
        "Shyvana",
        "MasterYi",
        "Kayn",
        "Belveth",
        "Lillia",
        "Diana",
        "Mordekaiser",
        "Mundo",
        "Amumu",
        "Rammus",
        "Skarner",
        "Nunu",
        "Ivern",
        "Zac",
        "Maokai",
        "Poppy",
        "TahmKench",
        "Sion",
        "ChoGath",
        "Malphite",
        "Nasus",
        "Tryndamere",
        "Jax",
        "Wukong",
        "Yorick",
        "Trundle",
        "Udyr",
        "Volibear",
    ],
    
    # Power Farmer - Fast clear speed, scales well with gold/XP
    "power_farmer": [
        "Graves",
        "Kindred",
        "Karthus",
        "Shyvana",
        "MasterYi",
        "Belveth",
        "Lillia",
        "Kayn",
        "Diana",
        "Taliyah",
        "Nidalee",
    ],
    
    # Invader - Champions that excel at counter-jungling
    "invader": [
        "Graves",
        "Nidalee",
        "Kindred",
        "KhaZix",
        "Rengar",
        "Elise",
        "LeeSin",
        "Shaco",
        "Taliyah",
    ],
    
    # Tank/Utility - Champions that provide CC and tankiness
    "tank_utility": [
        "Sejuani",
        "Amumu",
        "Zac",
        "Rammus",
        "Skarner",
        "Nunu",
        "Maokai",
        "Poppy",
        "TahmKench",
        "Sion",
        "ChoGath",
        "Malphite",
        "Volibear",
        "Udyr",
    ],
    
    # Assassin - High burst damage, target elimination
    "assassin": [
        "KhaZix",
        "Rengar",
        "Evelynn",
        "Shaco",
        "Talon",
        "Qiyana",
        "Zed",
        "Nocturne",
        "Ekko",
        "Fizz",
        "Kayn",
    ],
}

# Reverse mapping: champion name -> list of categories
CHAMPION_TO_CATEGORIES = {}

# Build reverse mapping
for category, champions in CHAMPION_CATEGORIES.items():
    for champion in champions:
        if champion not in CHAMPION_TO_CATEGORIES:
            CHAMPION_TO_CATEGORIES[champion] = []
        CHAMPION_TO_CATEGORIES[champion].append(category)

def get_champion_category(champion_name: str, primary: bool = False) -> str | list[str]:
    """
    Get the category/categories for a champion.
    
    Args:
        champion_name: The champion name (e.g., "LeeSin", "Fiddlesticks")
        primary: If True, returns only the first/primary category. If False, returns all categories.
    
    Returns:
        If primary=True: Single category string
        If primary=False: List of category strings
    """
    categories = CHAMPION_TO_CATEGORIES.get(champion_name, [])
    
    if not categories:
        # Default fallback - could be improved with better categorization
        return "aggressive" if primary else ["aggressive"]
    
    if primary:
        # Return the first category (could be improved with priority system)
        return categories[0]
    
    return categories

def get_champions_by_category(category: str) -> list[str]:
    """
    Get all champions in a specific category.
    
    Args:
        category: The category name (e.g., "aggressive", "full_clear")
    
    Returns:
        List of champion names in that category
    """
    return CHAMPION_CATEGORIES.get(category, [])

def is_champion_in_category(champion_name: str, category: str) -> bool:
    """
    Check if a champion belongs to a specific category.
    
    Args:
        champion_name: The champion name
        category: The category to check
    
    Returns:
        True if champion is in the category, False otherwise
    """
    return category in CHAMPION_TO_CATEGORIES.get(champion_name, [])

if __name__ == "__main__":
    # Example usage
    print("Champion Categorization System")
    print("=" * 50)
    
    # Test examples from user
    print("\nExample Champions:")
    print(f"Lee Sin: {get_champion_category('LeeSin')}")
    print(f"Jarvan IV: {get_champion_category('JarvanIV')}")
    print(f"Fiddlesticks: {get_champion_category('Fiddlesticks')}")
    print(f"Karthus: {get_champion_category('Karthus')}")
    
    print("\nAggressive Champions:")
    aggressive_champs = get_champions_by_category("aggressive")
    print(f"Total: {len(aggressive_champs)}")
    print(f"Examples: {', '.join(aggressive_champs[:10])}")
    
    print("\nFull Clear Champions:")
    full_clear_champs = get_champions_by_category("full_clear")
    print(f"Total: {len(full_clear_champs)}")
    print(f"Examples: {', '.join(full_clear_champs[:10])}")
