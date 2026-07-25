import re

with open("server/collectors/LlmProbe.js") as f:
    content = f.read()

# For all conflicts, keep BOTH (upstream + stashed) where they add different things
# For the baseUrl conflicts, keep our version (localhost fix for local sparks)
pattern = r'<<<<<<< Updated upstream\n(.*?)=======\n(.*?)>>>>>>> Stashed changes'

def replace(m):
    ours = m.group(1).strip()
    theirs = m.group(2).strip()
    
    # For baseUrl conflicts - keep stashed (our localhost fix)
    if "baseUrl" in ours and "baseUrl" in theirs:
        return theirs
    
    # For the /metrics section - keep stashed (our VllmMetricsParser integration)
    if "metricsText" in theirs or "vllmTelemetry" in theirs:
        return theirs
    
    # For the telemetry fields section - keep stashed (our expanded fields)
    if "runningSlots" in theirs or "vllmTelemetry" in theirs:
        return theirs
    
    # Default: keep both
    return ours + "\n" + theirs

content = re.sub(pattern, replace, content, flags=re.DOTALL)

with open("server/collectors/LlmProbe.js", "w") as f:
    f.write(content)

# Verify no conflict markers remain
if "<<<<<<<" in content:
    print("WARNING: conflict markers remain!")
else:
    print("LlmProbe.js resolved - kept our localhost fix + VllmMetricsParser integration + expanded telemetry fields")