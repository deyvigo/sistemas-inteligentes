import json

# Load the JSON file with full path
json_path = r'C:\Users\RYZEN\Documents\Proyectos\sistemas-inteligentes\back\pictogram-model\pictogramasArasaac.json'
with open(json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f'Total pictograms: {len(data)}')
print('=' * 80)

# Search for 'saco' in keywords
results = []
for item in data:
    if '_id' not in item or 'keywords' not in item:
        continue
    
    pictogram_id = item['_id']
    categories = item.get('categories', [])
    tags = item.get('tags', [])
    
    # Check all keywords
    matched_keywords = []
    for kw in item.get('keywords', []):
        keyword = kw.get('keyword', '')
        if 'saco' in keyword.lower():
            matched_keywords.append({
                'keyword': keyword,
                'meaning': kw.get('meaning', 'No meaning provided'),
                'plural': kw.get('plural', ''),
                'type': kw.get('type', '')
            })
    
    if matched_keywords:
        results.append({
            'id': pictogram_id,
            'matched_keywords': matched_keywords,
            'categories': categories,
            'tags': tags
        })

# Print results
print(f'Found {len(results)} pictogram(s) with "saco" in keywords:\n')

for result in results:
    print(f'Pictogram ID: {result["id"]}')
    print(f'Categories: {result["categories"]}')
    print(f'Tags: {result["tags"]}')
    print('Matched Keywords:')
    for kw in result['matched_keywords']:
        print(f'  - Keyword: {kw["keyword"]}')
        meaning = kw['meaning']
        if len(meaning) > 200:
            print(f'    Meaning: {meaning[:200]}...')
        else:
            print(f'    Meaning: {meaning}')
        if kw['plural']:
            print(f'    Plural: {kw["plural"]}')
    print('-' * 80)

# Check specifically for 'saco de papas'
print('\n' + '=' * 80)
print('Searching for "saco de papas" (potato chips bag):')
print('=' * 80)

found_papas = False
for item in data:
    if '_id' not in item or 'keywords' not in item:
        continue
    
    for kw in item.get('keywords', []):
        keyword = kw.get('keyword', '').lower()
        if 'saco de papas' in keyword or ('saco' in keyword and 'papas' in keyword):
            print(f'Found! Pictogram ID: {item["_id"]}')
            print(f'Keyword: {kw.get("keyword", "")}')
            print(f'Meaning: {kw.get("meaning", "No meaning")}')
            print(f'Categories: {item.get("categories", [])}')
            print(f'Tags: {item.get("tags", [])}')
            found_papas = True
            break

if not found_papas:
    print('No specific pictogram found for "saco de papas"')
    print('\nChecking for related terms like "papas fritas", "chips", etc...')
    related_terms = ['papas', 'papas fritas', 'chips', 'patatas', 'patatas fritas']
    for item in data:
        if '_id' not in item or 'keywords' not in item:
            continue
        for kw in item.get('keywords', []):
            keyword = kw.get('keyword', '').lower()
            for term in related_terms:
                if term in keyword:
                    print(f'Related: ID {item["_id"]} - Keyword: {kw.get("keyword", "")}')
                    break
