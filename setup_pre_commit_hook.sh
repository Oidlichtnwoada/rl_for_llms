#!/bin/bash -e

# create the file
touch .git/hooks/pre-commit

# make it executable
chmod +x .git/hooks/pre-commit

# fill the file with the relevant content
cat << 'EOF' > .git/hooks/pre-commit
#!/bin/bash -e

# call the check repo script
./check_repo.sh
EOF
