import subprocess

print("🚀 Running all test suites...\n")

result = subprocess.run(["pytest", "tests", "-v"], capture_output=True, text=True)

print(result.stdout)

if result.returncode == 0:
    print("✅ All tests passed!")
else:
    print("❌ Some tests failed. Check above for details.")
