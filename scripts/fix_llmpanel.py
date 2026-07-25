import re

with open("src/components/SparkPage/LlmPanel.tsx") as f:
    content = f.read()

pattern = r'<<<<<<< Updated upstream\n(.*?)=======\n(.*?)>>>>>>> Stashed changes'

def replace(m):
    ours = m.group(1).strip()  # upstream (rebased)
    theirs = m.group(2).strip()  # stashed (our dual-axis changes)
    
    # Conflict 1: Props interface - keep BOTH (upstream's onRemovePort + our legacy props)
    if "onRemovePort" in ours and "onLlmPortChange" in theirs:
        return theirs  # our version has all props including onRemovePort
    
    # Conflict 2: Function signature - keep stashed (has all params)
    if "export function LlmPanel" in ours and "export function LlmPanel" in theirs:
        return theirs  # our version has all destructured params
    
    # Conflict 3: handleSavePort - keep stashed (has onLlmPortChange callback)
    if "onLlmPortChange" in theirs:
        return theirs
    
    # Default: keep both
    return ours + "\n" + theirs

content = re.sub(pattern, replace, content, flags=re.DOTALL)

# Now apply the dual-axis chart changes to genSeries and preSeries
# Add yAxis: "left" to genSeries
content = content.replace(
    'data: history.genTps,\n    area: true,\n  };',
    'data: history.genTps,\n    area: true,\n    yAxis: "left",\n  };'
)

# Add yAxis: "right" to preSeries
content = content.replace(
    'data: history.prefillTps,\n    area: false,\n  };',
    'data: history.prefillTps,\n    area: false,\n    yAxis: "right",\n  };'
)

# Update TelemetryChart call to add yMax=140 and yUnitRight
content = content.replace(
    'yUnit=""\n              yMin={0}',
    'yUnit=""\n              yUnitRight=""\n              yMin={0}\n              yMax={140}'
)

with open("src/components/SparkPage/LlmPanel.tsx", "w") as f:
    f.write(content)

if "<<<<<<<" in content:
    print("WARNING: conflict markers remain!")
else:
    print("LlmPanel.tsx resolved - merged props + dual-axis chart changes applied")
    print("yAxis left:", 'yAxis: "left"' in content)
    print("yAxis right:", 'yAxis: "right"' in content)
    print("yMax=140:", "yMax={140}" in content)