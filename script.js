function processOMR() {
    console.log("Button Clicked!");

    const fileInput = document.getElementById("omrUpload");
    const registrationNumber = document.getElementById("registrationNumber").value;
    const verificationDiv = document.getElementById("verification");
    const marksDiv = document.getElementById("marks");

    if (!fileInput.files.length) {
        alert("Please upload an OMR sheet.");
        return;
    }

    const formData = new FormData();
    formData.append("omrImage", fileInput.files[0]);  // Backend expects "omrImage"
    formData.append("registrationNumber", registrationNumber); 

    console.log("Sending request to backend...");

    fetch("http://127.0.0.1:5000/process_omr", {  // Ensure Flask is running
        method: "POST",
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        console.log("Response received:", data);
        if (data.error) {
            verificationDiv.innerHTML = <span style="color:red;">${data.error}</span>;
        } else {
            verificationDiv.innerHTML = <span style="color:green;">Verification Successful</span>;
            marksDiv.innerHTML = Marks Obtained: <strong>${data.score}</strong>;
        }
    })
    .catch(error => {
        console.error("Error:", error);
        verificationDiv.innerHTML = <span style="color:red;">Error processing OMR.</span>;
    });
}
