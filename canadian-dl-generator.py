import csv
import json
import random
import re
import string
import os

# Define the patterns for Canadian driver's licenses by province
dl_patterns = {
    "Alberta": r'\d{6}-\d{3}|\d{5,9}',
    "British Columbia": r'\d{7}',
    "Manitoba": r'[A-Z]{2}-?[A-Z]{2}-?[A-Z]{2}-?[A-Z]\d{3}[A-Z]{2}',
    "New Brunswick": r'\d{5,7}',
    "Newfoundland and Labrador": r'[A-Z]\d{9}',
    "Nova Scotia": r'[A-Z]{5}-?[0123]\d[01]\d{6}',  # Precise pattern per documentation
    "Ontario": r'[A-Z]\d{4}-?\d{5}\d[0156]\d[0123]\d',  # Exact pattern per Microsoft docs
    "Prince Edward Island": r'\d{5,6}',
    "Quebec": r'[A-Z]\d{12}',
    "Saskatchewan": r'\d{8}',
    "Northwest Territories": r'\d{6}',
    "Nunavut": r'\d{5,6}',
    "Yukon": r'\d{1,6}'
}

def random_letter():
    """Generate a random uppercase letter."""
    return random.choice(string.ascii_uppercase)

def generate_license(province):
    """Generate a random driver's license number for the given province."""
    if province == "Alberta":
        # Format: 123456-789 or 123456789
        if random.random() > 0.5:
            return f"{random.randint(100000, 999999)}-{random.randint(100, 999)}"
        else:
            return f"{random.randint(10000, 99999999)}"
    
    elif province == "British Columbia":
        # Format: 1234567
        return f"{random.randint(1000000, 9999999)}"
    
    elif province == "Manitoba":
        # Format: AB-CD-EF-G123HI
        part1 = f"{random_letter()}{random_letter()}"
        part2 = f"{random_letter()}{random_letter()}"
        part3 = f"{random_letter()}{random_letter()}"
        part4 = f"{random_letter()}{random.randint(100, 999)}{random_letter()}{random_letter()}"
        # Sometimes include hyphens, sometimes not
        if random.random() > 0.5:
            return f"{part1}-{part2}-{part3}-{part4}"
        else:
            return f"{part1}{part2}{part3}{part4}"
    
    elif province == "New Brunswick":
        # Format: 12345 to 1234567
        digits = random.randint(5, 7)
        return f"{random.randint(10**(digits-1), 10**digits - 1)}"
    
    elif province == "Newfoundland and Labrador":
        # Format: A123456789
        return f"{random_letter()}{random.randint(100000000, 999999999)}"
    
    elif province == "Nova Scotia":
        # Format based on documentation:
        # - five letters
        # - optional hyphen
        # - one digit; any of 0, 1, 2 or 3
        # - one digit
        # - one digit zero or one
        # - six digits
        letters = ''.join(random_letter() for _ in range(5))
        
        # Restricted first digit (0-3)
        first_digit = random.randint(0, 3)
        
        # Any digit for second position
        second_digit = random.randint(0, 9)
        
        # Restricted third digit (0-1)
        third_digit = random.randint(0, 1)
        
        # Six random digits for the end
        end_digits = ''.join(str(random.randint(0, 9)) for _ in range(6))
        
        # Format with or without hyphen
        if random.random() > 0.5:
            return f"{letters}-{first_digit}{second_digit}{third_digit}{end_digits}"
        else:
            return f"{letters}{first_digit}{second_digit}{third_digit}{end_digits}"
    
    elif province == "Ontario":
        # Format based on Microsoft documentation:
        # - one letter
        # - four digits
        # - optional hyphen
        # - five digits
        # - one digit
        # - one digit; any of 0, 1, 5, or 6
        # - one digit
        # - one digit; any of 0, 1, 2 or 3
        # - one digit
        letter = random_letter()
        first_part = f"{random.randint(1000, 9999)}"  # 4 digits
        second_part = f"{random.randint(10000, 99999)}"  # 5 digits
        
        # Special digits as per specification
        normal_digit1 = random.randint(0, 9)
        restricted_digit1 = random.choice([0, 1, 5, 6])  # Must be 0, 1, 5, or 6
        normal_digit2 = random.randint(0, 9)
        restricted_digit2 = random.randint(0, 3)  # Must be 0, 1, 2, or 3
        final_digit = random.randint(0, 9)
        
        # Structure the final 5 digits properly
        end_part = f"{normal_digit1}{restricted_digit1}{normal_digit2}{restricted_digit2}{final_digit}"
        
        # Format with or without hyphen
        if random.random() > 0.5:
            return f"{letter}{first_part}-{second_part}{end_part}"
        else:
            return f"{letter}{first_part}{second_part}{end_part}"
    
    elif province == "Prince Edward Island":
        # Format: 12345 or 123456
        if random.random() > 0.5:
            return f"{random.randint(10000, 99999)}"
        else:
            return f"{random.randint(100000, 999999)}"
    
    elif province == "Quebec":
        # Format: A123456789012
        return f"{random_letter()}{random.randint(100000000000, 999999999999)}"
    
    elif province == "Saskatchewan":
        # Format: 12345678
        return f"{random.randint(10000000, 99999999)}"
    
    elif province == "Northwest Territories":
        # Format: 123456
        return f"{random.randint(100000, 999999)}"
    
    elif province == "Nunavut":
        # Format: 12345 or 123456
        if random.random() > 0.5:
            return f"{random.randint(10000, 99999)}"
        else:
            return f"{random.randint(100000, 999999)}"
    
    elif province == "Yukon":
        # Format: 1 to 123456
        length = random.randint(1, 6)
        return f"{random.randint(1, 10**length - 1)}"
    
    return "Invalid province"

def generate_context_text(license_number, province):
    """Generate a realistic text context containing the license number."""
    contexts = [
        f"Driver's license number {license_number} issued in {province} was verified successfully.",
        f"Please confirm your {province} driver's license: {license_number}.",
        f"The customer provided {province} license {license_number} as identification.",
        f"Verification required for {province} DL #{license_number}.",
        f"According to our records, your {province} driver's license {license_number} expires next month.",
        f"We've updated your profile with your new {province} driver's license number: {license_number}.",
        f"Your application with {province} license {license_number} has been approved.",
        f"Could not validate {province} driver's license with ID {license_number}.",
        f"The {province} driving license {license_number} was suspended due to traffic violations.",
        f"Please provide a copy of your {province} driver's license {license_number} for our records."
    ]
    return random.choice(contexts)

def find_license_in_text(text, license_number):
    """Find the position of the license number in the text and return label data."""
    start = text.find(license_number)
    if start == -1:
        return None
    
    end = start + len(license_number)
    return {
        "start": start,
        "end": end,
        "text": license_number,
        "labels": ["driver_license"]
    }

def validate_license(license_number, province):
    """Verify that a generated license matches its province's pattern."""
    pattern = dl_patterns[province]
    return bool(re.fullmatch(pattern, license_number))

def generate_validation_data(num_samples=100, output_file="canadian_dl_validation_data.csv"):
    """Generate synthetic validation data for Canadian driver's licenses."""
    provinces = list(dl_patterns.keys())
    data = []
    
    for _ in range(num_samples):
        province = random.choice(provinces)
        license_number = generate_license(province)
        
        # Validate that our generated license matches the pattern
        if not validate_license(license_number, province):
            attempts = 1
            max_attempts = 3
            # Try up to 3 times to generate a valid license
            while attempts < max_attempts and not validate_license(license_number, province):
                license_number = generate_license(province)
                attempts += 1
            
            # If still invalid after attempts, skip this sample
            if not validate_license(license_number, province):
                print(f"Warning: Generated license {license_number} for {province} does not match pattern {dl_patterns[province]}")
                continue
            
        text = generate_context_text(license_number, province)
        label_data = find_license_in_text(text, license_number)
        
        if label_data:
            data.append({
                "text": text,
                "label": json.dumps([label_data]),
                "province": province,  # For reference
                "license": license_number  # For reference
            })
    
    try:
        # Try to write to CSV
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['text', 'label']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for row in data:
                writer.writerow({
                    'text': row['text'],
                    'label': row['label']
                })
        
        print(f"Generated {len(data)} validation samples in {output_file}")
    except PermissionError:
        print(f"Error: Permission denied when writing to {output_file}")
        print("Try running the script with administrator privileges or choose a different output location.")
        print("The data has been generated but was not written to the file.")
    except Exception as e:
        print(f"Error writing to CSV: {str(e)}")
        print("The data has been generated but was not written to the file.")
    
    return data

# Execute the data generation
if __name__ == "__main__":
    # Generate 200 samples and specify an alternate output location if needed
    num_samples = 200
    output_file = "canadian_dl_validation_data.csv"
    
    # Alternative locations in case of permission issues
    alternative_locations = [
        "./canadian_dl_validation_data.csv",  # Current directory with explicit path
        os.path.join(os.path.expanduser("~"), "canadian_dl_validation_data.csv"),  # Home directory
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "canadian_dl_validation_data.csv")  # Script directory
    ]
    
    # Try to generate with default location first
    try:
        data = generate_validation_data(num_samples=num_samples, output_file=output_file)
    except Exception as e:
        print(f"\nError with default output location: {str(e)}")
        print("Trying alternative locations...")
        
        # Try alternative locations
        for alt_location in alternative_locations:
            try:
                print(f"Attempting to write to: {alt_location}")
                data = generate_validation_data(num_samples=num_samples, output_file=alt_location)
                output_file = alt_location  # Update if successful
                print(f"Successfully wrote to: {alt_location}")
                break
            except Exception as e:
                print(f"Failed with error: {str(e)}")
        else:
            print("\nCould not write to any location. Data will be generated but not saved.")
            data = generate_validation_data(num_samples=num_samples, output_file=None)
    
    # Calculate success rate
    success_rate = len(data) / num_samples * 100
    print(f"\nGeneration success rate: {success_rate:.2f}% ({len(data)} valid out of {num_samples} attempts)")
    
    # Verify a sample license from each province for testing purposes
    print("\nVerifying sample licenses for each province:")
    for province in dl_patterns.keys():
        for _ in range(5):  # Try up to 5 times to get a valid sample
            sample = generate_license(province)
            if validate_license(sample, province):
                print(f"{province}: {sample} ✓")
                break
        else:
            print(f"{province}: Failed to generate valid sample after 5 attempts ✗")